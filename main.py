import sys
import time
import joblib
from datetime import datetime
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer

from src import config
from src.logger import DualLogger
from src.data.load import load_and_clean_data
from src.data.preprocess import split_data, remove_highly_correlated_features, scale_features, apply_smote, save_processed_data, load_processed_data
from src.models.train import train_baselines
from src.models.tune import tune_random_forest
from src.models.evaluate import evaluate_model
from src.models.mlp import train_fast_mlp
from src.models.export import save_pipeline
from src.models.mlp_tf import build_and_train_tf

# Hijack the terminal output to save to a file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"demo_run_{timestamp}.txt")

def main():
    # Start Master Timer
    pipeline_start = time.time()

    print("Starting Pattern Recognition Pipeline...")
    
    # Check if processed data already exists
    if (config.PROCESSED_DATA_DIR / "X_train_smote.parquet").exists():
        print("Found existing processed data. Bypassing data ingestion and SMOTE...")
        X_train_smote, y_train_smote, X_val, y_val, X_test, y_test = load_processed_data(config.PROCESSED_DATA_DIR)
        
        # WE MUST LOAD THE SCALER HERE to build the pipelines later
        scaler = joblib.load(config.PROCESSED_DATA_DIR / "scaler.joblib")
    
    else:
        print("Processed data not found. Running full preprocessing pipeline...")
        
        # 1. Ingest Data
        df = load_and_clean_data(config.RAW_DATA_DIR)

        # 2. Splitting
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, 
            temp_size=config.TEMP_SPLIT_SIZE, 
            val_test_size=config.VAL_TEST_SPLIT_SIZE, 
            random_state=config.RANDOM_STATE
        )
        
        # 3. Feature Selection
        X_train, X_val, X_test = remove_highly_correlated_features(
            X_train, X_val, X_test, 
            threshold=config.CORR_THRESHOLD
        )
        
        # 4. Normalization
        X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
        
        # Save the scaler immediately for the API
        scaler_path = config.PROCESSED_DATA_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"SUCCESS: Scaler saved to {scaler_path}")
        
        # 5. Class Balancing
        X_train_smote, y_train_smote = apply_smote(
            X_train, y_train, 
            random_state=config.RANDOM_STATE
        )
        
        # Save the processed data for future runs
        save_processed_data(
            X_train_smote, y_train_smote, X_val, y_val, X_test, y_test, 
            config.PROCESSED_DATA_DIR
        )

    # =====================================================================
    # BUILD THE MASTER PREPROCESSOR FOR SAVING PIPELINES
    # =====================================================================
    # Get the exact feature names the models expect
    expected_features = X_train_smote.columns.tolist()
    
    # Create a transformer that scales the expected features and drops everything else
    preprocessor = ColumnTransformer(
        transformers=[('scaler', scaler, expected_features)],
        remainder='drop' 
    )

    # =====================================================================
    # BASELINES
    # =====================================================================
    nb_model, dt_model = train_baselines(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    evaluate_model(nb_model, X_val, y_val, model_name="Naive Bayes Baseline")
    save_pipeline(nb_model, preprocessor, config.MODEL_SAVE_DIR, "naive_bayes_pipeline")
    
    evaluate_model(dt_model, X_val, y_val, model_name="Decision Tree Baseline")
    save_pipeline(dt_model, preprocessor, config.MODEL_SAVE_DIR, "decision_tree_pipeline")

    # =====================================================================
    # OPTIMIZED RANDOM FOREST
    # =====================================================================
    best_rf_model = tune_random_forest(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    evaluate_model(best_rf_model, X_val, y_val, model_name="Optimized Random Forest")
    save_pipeline(best_rf_model, preprocessor, config.MODEL_SAVE_DIR, "best_rf_pipeline")

    # =====================================================================
    # SCIKIT-LEARN FAST MLP
    # =====================================================================
    mlp_model = train_fast_mlp(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    evaluate_model(mlp_model, X_val, y_val, model_name="Scikit-Learn MLP Prototype")
    save_pipeline(mlp_model, preprocessor, config.MODEL_SAVE_DIR, "fast_mlp_pipeline")

    # =====================================================================
    # THE PROPOSAL ENSEMBLE (VotingClassifier Integration)
    # =====================================================================
    print("\nBuilding the Single-File Ensemble Wrapper...")

    ensemble_model = VotingClassifier(
        estimators=[
            ('random_forest', best_rf_model),
            ('scikit_mlp', mlp_model)
        ],
        voting='soft'
    )
    ensemble_model.fit(X_train_smote, y_train_smote)
    
    evaluate_model(ensemble_model, X_val, y_val, model_name="Proposal Ensemble (RF + Fast MLP)")
    save_pipeline(ensemble_model, preprocessor, config.MODEL_SAVE_DIR, "final_ensemble_pipeline")

    # =====================================================================
    # TENSORFLOW STRICT COMPLIANCE
    # =====================================================================
    print("\nTraining Strict TensorFlow MLP Architecture...")
    
    X_train_tf = np.asarray(X_train_smote)
    y_train_tf = np.asarray(y_train_smote)
    X_val_tf = np.asarray(X_val)
    y_val_tf = np.asarray(y_val)

    tf_model = build_and_train_tf(X_train_tf, y_train_tf, X_val_tf, y_val_tf, config.RANDOM_STATE)
    
    # Save the TF model natively (no pipeline wrapping needed for the strict compliance artifact)
    tf_save_path = config.MODEL_SAVE_DIR / "mlp_tf_model.keras"
    tf_model.save(tf_save_path)
    print(f"\n[DIAGNOSTIC] TensorFlow model successfully saved to {tf_save_path}")

    # Tiny adapter class to allow scikit-learn's evaluate_model to seamlessly read TensorFlow outputs
    class KerasScikitWrapper:
        def __init__(self, model):
            self.model = model
        def predict(self, X):
            return np.argmax(self.model.predict(X, verbose=0), axis=1)
        def predict_proba(self, X):
            return self.model.predict(X, verbose=0)

    # Evaluate using your native function
    evaluate_model(KerasScikitWrapper(tf_model), X_val_tf, y_val_tf, model_name="TensorFlow MLP Prototype")

    # =====================================================================
    # LINEAR SVM (PROFESSOR'S RECOMMENDATION)
    # =====================================================================
    print("\nTraining Linear SVM...")
    linear_svm_model = LinearSVC(dual="auto", random_state=config.RANDOM_STATE, max_iter=2000)
    linear_svm_model.fit(X_train_smote, y_train_smote)
    
    evaluate_model(linear_svm_model, X_val, y_val, model_name="Linear SVM")
    save_pipeline(linear_svm_model, preprocessor, config.MODEL_SAVE_DIR, "linear_svm_pipeline")

    # End Master Timer
    pipeline_end = time.time()
    total_mins, total_secs = divmod(pipeline_end - pipeline_start, 60)
    print(f"\n[DIAGNOSTIC] Total Pipeline Execution Time: {int(total_mins)}m {int(total_secs)}s")
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()
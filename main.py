import sys
import time
import joblib
from datetime import datetime
from src import config
from src.logger import DualLogger
from src.data.load import load_and_clean_data
from src.data.preprocess import split_data, remove_highly_correlated_features, scale_features, apply_smote, save_processed_data, load_processed_data
from src.models.train import train_baselines
from src.models.tune import tune_random_forest
from src.models.evaluate import evaluate_model
from src.models.mlp import train_fast_mlp
from src.models.export import save_model

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

    # 6. Baseline Training & Evaluation
    nb_model, dt_model = train_baselines(X_train_smote, y_train_smote, config.RANDOM_STATE)
    evaluate_model(nb_model, X_val, y_val, model_name="Naive Bayes Baseline")
    evaluate_model(dt_model, X_val, y_val, model_name="Decision Tree Baseline")

    # 7. Model Tuning (Happens exclusively on Training Data)
    best_rf_model = tune_random_forest(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    # 8. Save the model immediately after tuning!
    save_model(best_rf_model, config.MODEL_SAVE_DIR, "best_rf_model")

    # 9. Final Evaluation (Testing the tuned model on Unseen Data)
    evaluate_model(best_rf_model, X_val, y_val, model_name="Optimized Random Forest")

    # --- Train the Fast MLP ---
    # 1. Train it directly on the SMOTE data (no grid search required for the fast prototype)
    mlp_model = train_fast_mlp(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    # 2. Save it immediately for the demo
    save_model(mlp_model, config.MODEL_SAVE_DIR, "fast_mlp_model")
    
    # 3. Evaluate it against the unseen Validation set
    evaluate_model(mlp_model, X_val, y_val, model_name="Scikit-Learn MLP Prototype")

    # End Master Timer
    pipeline_end = time.time()
    total_mins, total_secs = divmod(pipeline_end - pipeline_start, 60)
    print(f"\n[DIAGNOSTIC] Total Pipeline Execution Time: {int(total_mins)}m {int(total_secs)}s")
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()
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
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from src.models.mlp_tf import build_and_train_tf

# Hijack the terminal output to save to a file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"demo_run_{timestamp}.txt")

# --- Helper function for logging ---
def print_metrics(name, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        roc = roc_auc_score(y_true, y_prob)

        print(f"\n--- Evaluating {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"ROC-AUC:   {roc}") 

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

    # Train the Fast MLP (scikit)
    # 1. Train it directly on the SMOTE data (no grid search required for the fast prototype)
    mlp_model = train_fast_mlp(X_train_smote, y_train_smote, config.RANDOM_STATE)
    
    # 2. Save it immediately for the demo
    save_model(mlp_model, config.MODEL_SAVE_DIR, "fast_mlp_model")
    
    # 3. Evaluate it against the unseen Validation set
    evaluate_model(mlp_model, X_val, y_val, model_name="Scikit-Learn MLP Prototype")

    # Evaluate the RF + Scikit-MLP Ensemble
    rf_probs = best_rf_model.predict_proba(X_val)[:, 1] # type: ignore
    mlp_probs = mlp_model.predict_proba(X_val)[:, 1] # type: ignore
    
    ensemble_probs = (rf_probs + mlp_probs) / 2.0 # type: ignore
    ensemble_preds = np.where(ensemble_probs >= 0.5, 1, 0)
    
    print_metrics("Proposal Ensemble (RF + Fast MLP)", y_val, ensemble_preds, ensemble_probs)

    print("\nBuilding the Single-File Ensemble Wrapper...")

    # 1. Wrap the models together. 
    # voting='soft' tells it to average the probabilities automatically!
    ensemble_model = VotingClassifier(
        estimators=[
            ('random_forest', best_rf_model),
            ('scikit_mlp', mlp_model)
        ],
        voting='soft'
    )

    # 2. You HAVE to fit the wrapper so it knows how to route the data
    # (Since the individual models are already trained, this is very fast)
    ensemble_model.fit(X_train_smote, y_train_smote)

    # 3. Save it as a single file!
    ensemble_path = config.MODEL_SAVE_DIR / "final_ensemble_model.joblib"
    joblib.dump(ensemble_model, ensemble_path)
    print(f"SUCCESS: Single ensemble saved to {ensemble_path}")

    # TensorFlow MLP (Strict Math Compliance)
    print("\nTraining Strict TensorFlow MLP Architecture...")
    # Use np.asarray() to safely convert everything for TensorFlow
    X_train_tf = np.asarray(X_train_smote)
    y_train_tf = np.asarray(y_train_smote)
    X_val_tf = np.asarray(X_val)
    y_val_tf = np.asarray(y_val)

    tf_model = build_and_train_tf(X_train_tf, y_train_tf, X_val_tf, y_val_tf, config.RANDOM_STATE)
    
    # Evaluate TensorFlow model
    tf_probs_full = tf_model.predict(X_val_tf) # Returns probabilities for both classes
    tf_probs = tf_probs_full[:, 1]             # Isolate "Malicious" class probabilities
    tf_preds = np.argmax(tf_probs_full, axis=1)# Convert to 0 or 1 based on highest probability
    
    print_metrics("TensorFlow MLP Prototype", y_val_tf, tf_preds, tf_probs)
    
    # Save the TF model
    tf_save_path = config.MODEL_SAVE_DIR / "mlp_tf_model.keras"
    tf_model.save(tf_save_path)
    print(f"\n[DIAGNOSTIC] TensorFlow model successfully saved to {tf_save_path}")

    # End Master Timer
    pipeline_end = time.time()
    total_mins, total_secs = divmod(pipeline_end - pipeline_start, 60)
    print(f"\n[DIAGNOSTIC] Total Pipeline Execution Time: {int(total_mins)}m {int(total_secs)}s")
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    main()
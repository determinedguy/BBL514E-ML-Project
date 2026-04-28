# evaluate_svm.py
import sys
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src import config
from src.logger import DualLogger

# --- Hijack the terminal output to save receipts to a file ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"svm_original_evaluation_{timestamp}.txt")

def print_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate False Positive Rate
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
    print("\n" + "="*50)
    print("STARTING ORIGINAL SVM EVALUATION (10% SPLIT)")
    print("="*50)
    
    start_time = time.time()

    print("\nLoading Processed SMOTE Data...")
    X_train_full = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_train_smote.parquet")
    y_train_full = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_train_smote.parquet")['Label']
    
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']

    print("\nDownsampling to 10% Stratified (Original Configuration)...")
    _, X_svm, _, y_svm = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.10, 
        stratify=y_train_full, 
        random_state=config.RANDOM_STATE
    )

    print(f"Training on {len(X_svm)} rows...")
    print("Training Support Vector Machine (RBF Kernel)... (This may take a few minutes)")
    
    # The original, unconstrained SVM model
    svm = SVC(kernel='rbf', random_state=config.RANDOM_STATE)
    svm.fit(X_svm, y_svm)
    
    print("\nPredicting on Validation Set...")
    svm_preds = svm.predict(X_val)
    # We still use decision_function here because it calculates the ROC-AUC instantly 
    # without needing the massive RAM overhead of probability=True
    svm_decision_scores = svm.decision_function(X_val) 
    
    print_metrics("SVM (RBF Kernel)", y_val, svm_preds, svm_decision_scores)

    pipeline_end = time.time()
    total_mins, total_secs = divmod(pipeline_end - start_time, 60)
    print(f"\n[DIAGNOSTIC] Total SVM Execution Time: {int(total_mins)}m {int(total_secs)}s")
    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    main()
import sys
import time
from datetime import datetime
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src import config
from src.logger import DualLogger

# Hijack terminal output to save receipts
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"linear_svm_evaluation_{timestamp}.txt")

def print_metrics(name, y_true, y_pred, y_decision):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    roc = roc_auc_score(y_true, y_decision)

    print(f"\n--- Evaluating {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {roc}") 

def main():
    print("\n" + "="*50)
    print("STARTING LINEAR SVM EVALUATION (100% DATASET)")
    print("="*50)
    
    start_time = time.time()

    print("\nLoading Processed SMOTE Data...")
    # Loading 100% of the training data this time!
    X_train_full = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_train_smote.parquet")
    y_train_full = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_train_smote.parquet")['Label']
    
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']

    print(f"\nTraining on ALL {len(X_train_full)} rows...")
    print("Training Linear Support Vector Machine... (This will be fast!)")
    
    svm = LinearSVC(dual="auto", random_state=config.RANDOM_STATE, max_iter=2000)
    svm.fit(X_train_full, y_train_full)
    
    print("\nPredicting on Validation Set...")
    svm_preds = svm.predict(X_val)
    
    # LinearSVC uses decision_function instead of predict_proba
    svm_decision_scores = svm.decision_function(X_val) 
    
    print_metrics("Linear SVM", y_val, svm_preds, svm_decision_scores)

    pipeline_end = time.time()
    total_mins, total_secs = divmod(pipeline_end - start_time, 60)
    print(f"\n[DIAGNOSTIC] Total Linear SVM Execution Time: {int(total_mins)}m {int(total_secs)}s")
    print("--- Evaluation Finished ---")

if __name__ == "__main__":
    main()
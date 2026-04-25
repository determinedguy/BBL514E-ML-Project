from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_eval, y_eval, model_name="Model"):
    print(f"\n--- Evaluating {model_name} ---")
    
    # Get predictions
    y_pred = model.predict(X_eval)
    
    # Get probabilities for ROC-AUC (assuming positive class is 'Malicious')
    # Some models (like basic SVM) might not support predict_proba by default, 
    # so we use a try-except block to handle it gracefully.
    try:
        y_prob = model.predict_proba(X_eval)[:, 1]
        roc_auc = roc_auc_score(y_eval, y_prob)
    except AttributeError:
        roc_auc = "N/A (predict_proba not supported)"
    
    # Calculate Core Metrics (scikit-learn defaults to pos_label=1)
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred)
    rec = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)

    # Update confusion matrix to look for 0 and 1
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()

    # Calculate False Positive Rate from Confusion Matrix
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"FPR:       {fpr:.4f}")
    print(f"ROC-AUC:   {roc_auc}")
    
    return acc, prec, rec, f1, fpr

def plot_confusion_matrix(y_eval, y_pred, model_name="Model"):
    cm = confusion_matrix(y_eval, y_pred, labels=['Benign', 'Malicious'])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()
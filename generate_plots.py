import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
from src import config

# Create a folder to save your publication-ready graphs
os.makedirs(config.PLOT_DIR, exist_ok=True)

# --- 1. Set global styling for academic papers ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def main():
    print("Loading Validation Data and Saved Models...")
    
    # Load Data (Parquet preserves your column names, which is great for Feature Importance!)
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']
    
    # Load Models
    rf_model = joblib.load(config.MODEL_SAVE_DIR / "best_rf_model.joblib")
    mlp_model = joblib.load(config.MODEL_SAVE_DIR / "fast_mlp_model.joblib")
    ensemble_model = joblib.load(config.MODEL_SAVE_DIR / "final_ensemble_model.joblib")

    print("\nGenerating Graph 1: Comparative ROC Curve...")
    plt.figure(figsize=(10, 8))
    
    models = {
        "Optimized Random Forest": rf_model,
        "Fast MLP (Scikit)": mlp_model,
        "Soft Voting Ensemble": ensemble_model
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for (name, model), color in zip(models.items(), colors):
        # Get probabilities for the "Malicious" class
        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    
    roc_path = config.PLOT_DIR / "comparative_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"-> Saved to {roc_path}")


    print("\nGenerating Graph 2: Confusion Matrix Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices on Validation Data', fontweight='bold', fontsize=16)
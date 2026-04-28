import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
from src import config

# Create a folder to save your publication-ready graphs
PLOT_DIR = config.PLOT_DIR
os.makedirs(PLOT_DIR, exist_ok=True)

# Set global styling for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

def main():
    print("Loading Validation Data and Saved Models...")
    
    # Load Data (Parquet preserves column names for Feature Importance)
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']
    
    # Load Models
    print("Loading model artifacts...")
    rf_model = joblib.load(config.MODEL_SAVE_DIR / "best_rf_model.joblib")
    mlp_model = joblib.load(config.MODEL_SAVE_DIR / "fast_mlp_model.joblib")
    ensemble_model = joblib.load(config.MODEL_SAVE_DIR / "final_ensemble_model.joblib")

    # =====================================================================
    # GRAPH 1: COMPARATIVE ROC CURVE
    # =====================================================================
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
    
    roc_path = PLOT_DIR / "comparative_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"-> Saved to {roc_path}")


    # =====================================================================
    # GRAPH 2: CONFUSION MATRIX HEATMAPS
    # =====================================================================
    print("\nGenerating Graph 2: Confusion Matrix Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices on Validation Data', fontweight='bold', fontsize=16)

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Benign', 'Malicious'], 
                    yticklabels=['Benign', 'Malicious'])
        
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    cm_path = PLOT_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"-> Saved to {cm_path}")


    # =====================================================================
    # GRAPH 3: FEATURE IMPORTANCE
    # =====================================================================
    print("\nGenerating Graph 3: Random Forest Feature Importance...")
    plt.figure(figsize=(12, 8))
    
    # Extract importances safely (in case it is wrapped in a pipeline or grid search)
    try:
        importances = rf_model.feature_importances_
    except AttributeError:
        # If it was saved as a GridSearchCV object, we need the best_estimator_
        importances = rf_model.best_estimator_.feature_importances_
    
    # Create a DataFrame to sort them easily
    feature_df = pd.DataFrame({
        'Feature': X_val.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top 20 features to keep the graph readable
    top_n = 20
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(top_n), palette='viridis')
    
    plt.title(f'Top {top_n} Most Critical Network Features (Random Forest)', fontweight='bold', fontsize=14)
    plt.xlabel('Relative Importance', fontweight='bold')
    plt.ylabel('Feature Name', fontweight='bold')
    
    feat_path = PLOT_DIR / "feature_importance.png"
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"-> Saved to {feat_path}")

    print("\nSUCCESS: All 3 publication-ready graphs have been generated in the /plots/ directory!")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
import tensorflow as tf
from src import config

# Create a folder to save your publication-ready graphs
PLOT_DIR = config.PLOT_DIR
os.makedirs(PLOT_DIR, exist_ok=True)

# Set global styling for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Tiny adapter class to allow scikit-learn metrics to seamlessly read TensorFlow outputs
class KerasScikitWrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)
    def predict_proba(self, X):
        return self.model.predict(X, verbose=0)

def main():
    print("Loading Validation Data and Saved Models...")
    
    # Load Data (Parquet preserves column names)
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']
    
    # Convert to numpy array safely for TensorFlow
    X_val_np = np.asarray(X_val)

    # Load Models
    print("Loading Scikit-Learn artifacts...")
    rf_model = joblib.load(config.MODEL_SAVE_DIR / "best_rf_model.joblib")
    mlp_model = joblib.load(config.MODEL_SAVE_DIR / "fast_mlp_model.joblib")
    ensemble_model = joblib.load(config.MODEL_SAVE_DIR / "final_ensemble_model.joblib")
    
    print("Loading TensorFlow artifact...")
    tf_keras_model = tf.keras.models.load_model(config.MODEL_SAVE_DIR / "mlp_tf_model.keras")
    tf_model_wrapped = KerasScikitWrapper(tf_keras_model)

    # Dictionary of all models for easy plotting
    models = {
        "Optimized Random Forest": rf_model,
        "Fast MLP (Scikit)": mlp_model,
        "Soft Voting Ensemble": ensemble_model,
        "Strict TensorFlow MLP": tf_model_wrapped
    }

    # =====================================================================
    # GRAPH 1: COMPARATIVE ROC CURVE (Full & Zoomed)
    # =====================================================================
    print("\nGenerating Graph 1: Comparative ROC Curve...")
    
    # Create a 1x2 side-by-side layout for the ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold', fontsize=16)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_val_np)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot on the Full View (ax1)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        # Plot on the Zoomed View (ax2)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')

    # Formatting: Full View
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.6)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
    ax1.set_title('Full View', fontweight='bold')
    ax1.legend(loc="lower right")

    # Formatting: Zoomed View (Magnifying the top-left corner)
    ax2.set_xlim([-0.001, 0.05]) # Focus strictly on 0% to 5% FPR
    ax2.set_ylim([0.95, 1.001])  # Focus strictly on 95% to 100% TPR
    ax2.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
    ax2.set_title('Zoomed View (Top-Left)', fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    
    roc_path = PLOT_DIR / "comparative_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved to {roc_path}")

    # =====================================================================
    # GRAPH 2: CONFUSION MATRIX HEATMAPS
    # =====================================================================
    print("\nGenerating Graph 2: Confusion Matrix Heatmaps...")
    # Changed from 1x3 to 2x2 grid to fit 4 models beautifully
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Confusion Matrices on Validation Data', fontweight='bold', fontsize=16)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_val_np)
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
    # GRAPH 3: FEATURE IMPORTANCE (Random Forest Only)
    # =====================================================================
    print("\nGenerating Graph 3: Random Forest Feature Importance...")
    plt.figure(figsize=(12, 8))
    
    # Extract importances
    try:
        importances = rf_model.feature_importances_
    except AttributeError:
        importances = rf_model.best_estimator_.feature_importances_
    
    feature_df = pd.DataFrame({
        'Feature': X_val.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    top_n = 20
    sns.barplot(x='Importance', y='Feature', data=feature_df.head(top_n), palette='viridis', hue='Feature', legend=False)
    
    plt.title(f'Top {top_n} Most Critical Network Features (Random Forest)', fontweight='bold', fontsize=14)
    plt.xlabel('Relative Importance', fontweight='bold')
    plt.ylabel('Feature Name', fontweight='bold')
    
    feat_path = PLOT_DIR / "feature_importance.png"
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"-> Saved to {feat_path}")

    print("\nSUCCESS: All publication-ready graphs have been generated in the /plots/ directory!")

if __name__ == "__main__":
    main()
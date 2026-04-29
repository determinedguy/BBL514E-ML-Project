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
os.makedirs(config.PLOT_DIR, exist_ok=True)

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
    print("Loading Validation Data...")
    
    # Load Data (Parquet preserves column names)
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']
    
    # Convert to numpy array safely for TensorFlow
    X_val_np = np.asarray(X_val)

    print("Loading Scikit-Learn Pipelines and Extracting Classifiers...")
    # We extract .named_steps['classifier'] to avoid double-scaling the already processed X_val data!
    nb_model = joblib.load(config.MODEL_SAVE_DIR / "naive_bayes_pipeline.joblib").named_steps['clf']
    dt_model = joblib.load(config.MODEL_SAVE_DIR / "decision_tree_pipeline.joblib").named_steps['clf']
    rf_model = joblib.load(config.MODEL_SAVE_DIR / "best_rf_pipeline.joblib").named_steps['clf']
    mlp_model = joblib.load(config.MODEL_SAVE_DIR / "fast_mlp_pipeline.joblib").named_steps['clf']
    ensemble_model = joblib.load(config.MODEL_SAVE_DIR / "final_ensemble_pipeline.joblib").named_steps['clf']
    
    print("Loading TensorFlow artifact...")
    tf_keras_model = tf.keras.models.load_model(config.MODEL_SAVE_DIR / "mlp_tf_model.keras")
    tf_model_wrapped = KerasScikitWrapper(tf_keras_model)

    # Dictionary of all 6 models!
    models = {
        "Naive Bayes": nb_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "Scikit MLP": mlp_model,
        "TensorFlow MLP": tf_model_wrapped,
        "Ensemble (RF + MLP)": ensemble_model
    }

    # =====================================================================
    # GRAPH 1: COMPARATIVE ROC CURVE (Full & Zoomed)
    # =====================================================================
    print("\nGenerating Graph 1: Comparative ROC Curve (6 Models)...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Receiver Operating Characteristic (ROC) Comparison', fontweight='bold', fontsize=16)
    
    # 6 distinct colors for the 6 models
    colors = ['#9467bd', '#8c564b', '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_val_np)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
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
    ax2.set_xlim([-0.001, 0.05]) 
    ax2.set_ylim([0.95, 1.001])  
    ax2.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
    ax2.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
    ax2.set_title('Zoomed View (Top-Left)', fontweight='bold')
    ax2.legend(loc="lower right")
    ax2.grid(True, which='both', linestyle='--', alpha=0.4)
    
    roc_path = config.PLOT_DIR / "comparative_roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved to {roc_path}")

    # =====================================================================
    # GRAPH 2: CONFUSION MATRIX HEATMAPS (2x3 Grid)
    # =====================================================================
    print("\nGenerating Graph 2: Confusion Matrix Heatmaps (6 Models)...")
    
    # Changed from 2x2 to a 2x3 grid to fit all 6 models perfectly!
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Confusion Matrices on Validation Data', fontweight='bold', fontsize=16)
    
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

    cm_path = config.PLOT_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved to {cm_path}")

    # =====================================================================
    # GRAPH 3: FEATURE IMPORTANCE (Random Forest Only)
    # =====================================================================
    print("\nGenerating Graph 3: Random Forest Feature Importance...")
    plt.figure(figsize=(12, 8))
    
    # Extract importances directly from the un-pipelined Random Forest
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
    
    feat_path = config.PLOT_DIR / "feature_importance.png"
    plt.savefig(feat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Saved to {feat_path}")

    print("\nSUCCESS: All publication-ready graphs have been generated in the /plots/ directory!")

if __name__ == "__main__":
    main()
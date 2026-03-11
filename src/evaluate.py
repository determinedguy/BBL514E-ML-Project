# src/evaluate.py
"""
Evaluate and compare trained models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score)
from .config import MODELS_DIR, RESULTS_DIR

def load_models():
    """Load all trained models"""
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)
    
    models = {}
    model_files = {
        'Random Forest': 'random_forest.pkl',
        'MLP': 'mlp.pkl',
        'SVM': 'svm.pkl',
        'KNN': 'knn.pkl',
        'Ensemble': 'ensemble.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"✓ Loaded: {name}")
        else:
            print(f"⚠ Not found: {name}")
    
    return models

def load_test_data():
    """Load test data and label encoder"""
    from .config import DATA_PROCESSED_DIR
    
    test_path = os.path.join(DATA_PROCESSED_DIR, 'test_data.pkl')
    le_path = os.path.join(DATA_PROCESSED_DIR, 'label_encoder.pkl')
    
    test_data = joblib.load(test_path)
    le = joblib.load(le_path)
    
    X_test = test_data['X']
    y_test = test_data['y']
    
    return X_test, y_test, le

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print('='*60)
    
    # Prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Try to calculate ROC-AUC (requires probability predictions)
    try:
        y_pred_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Prediction time: {pred_time:.4f} seconds")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pred_time': pred_time,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

def compare_models(results):
    """Create comparison visualizations"""
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Prepare data for plotting
    models = [r['model_name'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # 1. Metrics comparison (bar chart)
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR}/model_comparison.png")
    plt.close()
    
    # 2. Performance table
    df_results = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}",
            'ROC-AUC': f"{r['roc_auc']:.4f}" if r['roc_auc'] else 'N/A',
            'Pred Time (s)': f"{r['pred_time']:.4f}"
        }
        for r in results
    ])
    
    # Save as CSV
    csv_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Print table
    print("\n" + "="*60)
    print("MODEL PERFORMANCE TABLE")
    print("="*60)
    print(df_results.to_string(index=False))
    
    # 3. Prediction time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_times = [r['pred_time'] for r in results]
    colors = sns.color_palette('husl', len(models))
    ax.barh(models, pred_times, color=colors, edgecolor='black')
    ax.set_xlabel('Prediction Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_time.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR}/prediction_time.png")
    plt.close()

def plot_confusion_matrices(results, label_encoder):
    """Plot confusion matrices for all models"""
    print("\n" + "="*60)
    print("CREATING CONFUSION MATRICES")
    print("="*60)
    
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    class_names = label_encoder.classes_
    
    for idx, result in enumerate(results):
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[idx], cbar=True)
        axes[idx].set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.4f}", 
                          fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR}/confusion_matrices.png")
    plt.close()

def evaluate_all_models():
    """Evaluate all trained models"""
    print("\n" + "="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    # Load models and data
    models = load_models()
    X_test, y_test, le = load_test_data()
    
    if not models:
        print("\n✗ No models found! Train models first.")
        return
    
    print(f"\n✓ Loaded {len(models)} models")
    print(f"✓ Test set size: {X_test.shape[0]} samples")
    
    # Evaluate each model
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
    
    # Compare models
    compare_models(results)
    
    # Plot confusion matrices
    plot_confusion_matrices(results, le)
    
    # Find best model
    best_model = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*60)
    print("BEST MODEL")
    print("="*60)
    print(f"Model: {best_model['model_name']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"F1-Score: {best_model['f1_score']:.4f}")
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}/")
    
    return results
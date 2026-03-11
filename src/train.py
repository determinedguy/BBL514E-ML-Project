# src/train.py
"""
Train machine learning models
"""

import numpy as np
import joblib
import os
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from .config import (MODELS_DIR, RF_PARAMS, MLP_PARAMS, SVM_PARAMS, KNN_PARAMS)

def load_processed_data():
    """Load preprocessed data"""
    print("\n" + "="*60)
    print("LOADING PROCESSED DATA")
    print("="*60)
    
    from .config import DATA_PROCESSED_DIR
    
    train_path = os.path.join(DATA_PROCESSED_DIR, 'train_data.pkl')
    test_path = os.path.join(DATA_PROCESSED_DIR, 'test_data.pkl')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Processed data not found in {DATA_PROCESSED_DIR}. Run preprocessing first!"
        )
    
    train_data = joblib.load(train_path)
    test_data = joblib.load(test_path)
    
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    y_test = test_data['y']
    
    print(f"\n✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Number of features: {X_train.shape[1]}")
    print(f"✓ Number of classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    print(f"\nParameters: {RF_PARAMS}")
    
    start_time = time.time()
    
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'random_forest.pkl')
    joblib.dump(rf, model_path)
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    return rf, train_time

def train_mlp(X_train, y_train):
    """Train Multi-Layer Perceptron"""
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK (MLP)")
    print("="*60)
    print(f"\nParameters: {MLP_PARAMS}")
    
    start_time = time.time()
    
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'mlp.pkl')
    joblib.dump(mlp, model_path)
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    return mlp, train_time

def train_svm(X_train, y_train, sample_fraction=0.3):
    """Train SVM with RBF kernel (on sampled data for speed)"""
    print("\n" + "="*60)
    print("TRAINING SVM WITH RBF KERNEL")
    print("="*60)
    print(f"\n⚠ Warning: SVM can be slow. Using {sample_fraction*100:.0f}% of training data")
    
    # Sample data for faster training
    n_samples = int(len(X_train) * sample_fraction)
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    print(f"Training on {n_samples} samples (out of {len(X_train)})")
    print(f"\nParameters: {SVM_PARAMS}")
    
    start_time = time.time()
    
    svm = SVC(**SVM_PARAMS)
    svm.fit(X_train_sample, y_train_sample)
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'svm.pkl')
    joblib.dump(svm, model_path)
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    return svm, train_time

def train_knn(X_train, y_train, sample_fraction=0.2):
    """Train K-Nearest Neighbors (on sampled data)"""
    print("\n" + "="*60)
    print("TRAINING K-NEAREST NEIGHBORS")
    print("="*60)
    print(f"\n⚠ Warning: KNN stores all training data. Using {sample_fraction*100:.0f}% for efficiency")
    
    # Sample data
    n_samples = int(len(X_train) * sample_fraction)
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    print(f"Training on {n_samples} samples (out of {len(X_train)})")
    print(f"\nParameters: {KNN_PARAMS}")
    
    start_time = time.time()
    
    knn = KNeighborsClassifier(**KNN_PARAMS)
    knn.fit(X_train_sample, y_train_sample)
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'knn.pkl')
    joblib.dump(knn, model_path)
    
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    return knn, train_time

def train_ensemble(X_train, y_train):
    """Train ensemble model (voting classifier)"""
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE MODEL")
    print("="*60)
    
    # Load individual models
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp.pkl'))
    
    print("\nCombining Random Forest + MLP")
    print("Voting strategy: Soft voting (probability-based)")
    
    start_time = time.time()
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('mlp', mlp)
        ],
        voting='soft'
    )
    
    # Fit on full data (models are already trained, just combining)
    ensemble.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'ensemble.pkl')
    joblib.dump(ensemble, model_path)
    
    print(f"\n✓ Ensemble created in {train_time:.2f} seconds")
    print(f"✓ Model saved to: {model_path}")
    
    return ensemble, train_time

def train_all_models():
    """Train all models"""
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    results = {}
    
    # Train Random Forest
    rf, rf_time = train_random_forest(X_train, y_train)
    results['Random Forest'] = {'model': rf, 'train_time': rf_time}
    
    # Train MLP
    mlp, mlp_time = train_mlp(X_train, y_train)
    results['MLP'] = {'model': mlp, 'train_time': mlp_time}
    
    # Train SVM (optional - can be slow)
    try:
        svm, svm_time = train_svm(X_train, y_train)
        results['SVM'] = {'model': svm, 'train_time': svm_time}
    except Exception as e:
        print(f"\n✗ SVM training failed: {e}")
    
    # Train KNN (optional)
    try:
        knn, knn_time = train_knn(X_train, y_train)
        results['KNN'] = {'model': knn, 'train_time': knn_time}
    except Exception as e:
        print(f"\n✗ KNN training failed: {e}")
    
    # Train Ensemble
    ensemble, ens_time = train_ensemble(X_train, y_train)
    results['Ensemble'] = {'model': ensemble, 'train_time': ens_time}
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for name, info in results.items():
        print(f"{name:20s}: {info['train_time']:8.2f} seconds")
    
    print("\n" + "="*60)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*60)
    
    return results
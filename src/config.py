# src/config.py
"""
Project configuration and constants
"""
import os
import glob

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UPDATED: Point to your actual data location
DATA_DIR = '/teamspace/lightning_storage/MachineLearningCVE'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data_processed')  # NEW: For processed data

# Create directories if they don't exist (but NOT data dir - it already exists)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# AUTO-DETECT: Get all CSV files from data directory
DATA_FILES = [os.path.basename(f) for f in glob.glob(os.path.join(DATA_DIR, '*.csv'))]

# Sort for consistency
DATA_FILES.sort()

# Print detected files (for debugging)
if DATA_FILES:
    print(f"\n✓ Detected {len(DATA_FILES)} CSV files in {DATA_DIR}")
else:
    print(f"\n⚠ Warning: No CSV files found in {DATA_DIR}")

# Processed data filenames (stored in data_processed/)
PROCESSED_TRAIN = 'train_data.pkl'
PROCESSED_TEST = 'test_data.pkl'
SCALER_FILE = 'scaler.pkl'
LABEL_ENCODER_FILE = 'label_encoder.pkl'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 500000  # Use subset for faster iteration

# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': 1
}

MLP_PARAMS = {
    'hidden_layer_sizes': (128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 100,
    'random_state': RANDOM_STATE,
    'verbose': True,
    'early_stopping': True
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'gamma': 'scale',
    'probability': True,
    'random_state': RANDOM_STATE,
    'verbose': True
}

KNN_PARAMS = {
    'n_neighbors': 5,
    'n_jobs': -1,
    'algorithm': 'auto'
}

# Features to drop (non-predictive)
FEATURES_TO_DROP = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']

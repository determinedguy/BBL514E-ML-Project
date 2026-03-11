# src/predict.py
"""
Prediction module - classify new network traffic data
"""

import pandas as pd
import numpy as np
import joblib
import os
from .config import MODELS_DIR, DATA_PROCESSED_DIR

def load_trained_model(model_name='ensemble'):
    """Load a trained model"""
    model_files = {
        'rf': 'random_forest.pkl',
        'random_forest': 'random_forest.pkl',
        'mlp': 'mlp.pkl',
        'neural_network': 'mlp.pkl',
        'svm': 'svm.pkl',
        'knn': 'knn.pkl',
        'ensemble': 'ensemble.pkl'
    }
    
    model_file = model_files.get(model_name.lower(), 'ensemble.pkl')
    model_path = os.path.join(MODELS_DIR, model_file)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first!")
    
    print(f"\n✓ Loading model: {model_file}")
    model = joblib.load(model_path)
    return model

def load_preprocessing_artifacts():
    """Load scaler and label encoder"""
    scaler_path = os.path.join(DATA_PROCESSED_DIR, 'scaler.pkl')
    le_path = os.path.join(DATA_PROCESSED_DIR, 'label_encoder.pkl')
    
    if not os.path.exists(scaler_path) or not os.path.exists(le_path):
        raise FileNotFoundError("Preprocessing artifacts not found. Run preprocessing first!")
    
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(le_path)
    
    print(f"✓ Loaded scaler and label encoder")
    
    return scaler, label_encoder

def preprocess_input_data(df, scaler):
    """Preprocess input data for prediction"""
    print("\n" + "="*60)
    print("PREPROCESSING INPUT DATA")
    print("="*60)
    
    df = df.copy()
    
    print(f"Input shape: {df.shape}")
    
    # Remove Label column if present
    if 'Label' in df.columns:
        print("⚠ Removing 'Label' column from input")
        df = df.drop('Label', axis=1)
    
    # Handle missing values
    print("\nHandling missing values...")
    df = df.fillna(df.median(numeric_only=True))
    
    # Handle infinity values
    print("Handling infinity values...")
    df = df.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    print(f"Preprocessed shape: {df.shape}")
    
    # Scale features
    print("\nScaling features...")
    X_scaled = scaler.transform(df)
    
    print(f"✓ Data ready for prediction: {X_scaled.shape}")
    
    return X_scaled

def predict_from_csv(csv_path, model_name='ensemble'):
    """
    Predict attack types from a CSV file
    
    Args:
        csv_path: Path to CSV file with network flow features
        model_name: Model to use ('rf', 'mlp', 'svm', 'knn', 'ensemble')
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*60)
    print("PREDICTION MODE - CSV INPUT")
    print("="*60)
    print(f"Input file: {csv_path}")
    print(f"Model: {model_name}")
    
    # Load input data
    print("\nLoading input data...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Load model and preprocessing artifacts
    model = load_trained_model(model_name)
    scaler, label_encoder = load_preprocessing_artifacts()
    
    # Preprocess
    X = preprocess_input_data(df, scaler)
    
    # Predict
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    predictions_encoded = model.predict(X)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Get prediction probabilities if available
    try:
        probabilities = model.predict_proba(X)
        confidence = np.max(probabilities, axis=1)
    except:
        confidence = None
    
    # Create results dataframe
    results = pd.DataFrame({
        'Prediction': predictions,
        'Prediction_Code': predictions_encoded
    })
    
    if confidence is not None:
        results['Confidence'] = confidence
    
    print(f"\n✓ Predictions complete!")
    
    # Summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(results['Prediction'].value_counts())
    
    return results

def predict_from_pcap(pcap_path, model_name='ensemble'):
    """
    Predict attack types from a PCAP file
    (Requires CICFlowMeter to extract features first)
    
    Args:
        pcap_path: Path to PCAP file
        model_name: Model to use
    
    Returns:
        DataFrame with predictions
    """
    print("\n" + "="*60)
    print("PREDICTION MODE - PCAP INPUT")
    print("="*60)
    print(f"Input file: {pcap_path}")
    
    print("\n⚠ PCAP processing requires CICFlowMeter")
    print("This feature extracts flow features from PCAP files")
    print("\nFor now, please convert your PCAP to CSV using CICFlowMeter first,")
    print("then use predict_from_csv() instead.")
    print("\nAlternatively, you can:")
    print("1. Install cicflowmeter: pip install cicflowmeter")
    print("2. Run: cicflowmeter -f input.pcap -c output.csv")
    print("3. Use the output CSV with this tool")
    
    raise NotImplementedError("PCAP processing not yet implemented. Convert to CSV first.")

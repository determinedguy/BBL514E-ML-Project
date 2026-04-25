import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# --- 1. Initialization (Run once when the server starts) ---
print("Loading Pattern Recognition Models into memory...")

# Update these paths based on where the backend server files are located
MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

# Load the saved artifacts
scaler = joblib.load(DATA_DIR / "scaler.joblib")
rf_model = joblib.load(MODEL_DIR / "best_rf_model.joblib")
mlp_model = joblib.load(MODEL_DIR / "fast_mlp_model.joblib")

# Keep the exact feature columns your models expect
EXPECTED_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# --- 2. The Prediction Route (Run every time a user uploads a CSV) ---
def predict_traffic(csv_file_path: str):
    # 1. Load the new, unseen data
    live_data = pd.read_csv(csv_file_path)
    
    # Ensure columns match exactly (drop anything extra like an old 'Label' column)
    X_live = live_data[EXPECTED_FEATURES]
    
    # 2. Apply the EXACT same scaling logic from your training phase
    X_live_scaled = scaler.transform(X_live)
    
    # 3. Get probabilities from both models (Node 1 is the "Malicious" probability)
    rf_probs = rf_model.predict_proba(X_live_scaled)[:, 1]
    mlp_probs = mlp_model.predict_proba(X_live_scaled)[:, 1]
    
    # 4. ENSEMBLE: Average the probabilities (Soft Voting)
    ensemble_probs = (rf_probs + mlp_probs) / 2.0
    
    # 5. Convert probabilities back into final labels (Threshold = 0.5)
    predictions = np.where(ensemble_probs >= 0.5, 1, 0)
    
    # Append results to the dataframe for Haluk to display
    live_data['Ensemble_Probability'] = ensemble_probs
    live_data['Final_Prediction'] = ["Malicious" if p == 1 else "Benign" for p in predictions]
    
    return live_data[['Destination Port', 'Ensemble_Probability', 'Final_Prediction']]

# Example usage for testing:
# print(predict_traffic("path/to/uploaded_file.csv"))
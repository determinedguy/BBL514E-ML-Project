import sys
from datetime import datetime
import pandas as pd
from src import config
from src.logger import DualLogger
from src.models.mlp_tf import build_and_train_tf

# Hijack the terminal output to save to a file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = DualLogger(f"overnight_build_{timestamp}.txt")

def run_overnight_build():
    print("Loading datasets from disk...")
    
    X_train = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_train_smote.parquet")
    y_train = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_train_smote.parquet")['Label']
    
    X_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "X_val.parquet")
    y_val = pd.read_parquet(config.PROCESSED_DATA_DIR / "y_val.parquet")['Label']
    
    tf_model = build_and_train_tf(
        X_train.values, y_train.values, 
        X_val.values, y_val.values, 
        config.RANDOM_STATE
    )
    
    # Save it natively as a Keras file
    save_path = config.MODEL_SAVE_DIR / "mlp_tf_model.keras"
    tf_model.save(save_path)
    print(f"\nSUCCESS: Formal TensorFlow model saved to {save_path}")

if __name__ == "__main__":
    run_overnight_build()
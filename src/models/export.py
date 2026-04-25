import joblib
import os
from pathlib import Path

def save_model(model, save_dir: Path, model_name: str):
    print(f"Saving {model_name} to disk...")
    os.makedirs(save_dir, exist_ok=True)
    
    file_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model, file_path)
    print(f"Successfully saved to {file_path}")

def load_model(file_path: Path):
    print(f"Loading model from {file_path}...")
    return joblib.load(file_path)
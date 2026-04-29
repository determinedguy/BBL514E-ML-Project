import joblib
from sklearn.pipeline import Pipeline

def save_model(model, save_dir, filename):
    """Saves a raw model without any preprocessing steps attached."""
    save_path = save_dir / f"{filename}.joblib"
    joblib.dump(model, save_path)
    print(f"SUCCESS: Model saved to {save_path}")

def save_pipeline(model, preprocessor, save_dir, filename):
    """
    Wraps a preprocessor and a trained model into a Scikit-Learn Pipeline 
    to prevent feature mismatch in production, then saves it to disk.
    """
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    save_path = save_dir / f"{filename}.joblib"
    joblib.dump(pipe, save_path)
    print(f"SUCCESS: Production pipeline saved to {save_path}")
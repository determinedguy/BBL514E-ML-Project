import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def split_data(df: pd.DataFrame, temp_size: float, val_test_size: float, random_state: int, target_col: str = 'Label'):
    print("Splitting data into Train, Validation, and Test sets...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: Train vs Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    
    # Second split: Divide Temp into Val and Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_size, stratify=y_temp, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def remove_highly_correlated_features(X_train, X_val, X_test, threshold: float):
    print(f"Removing features with correlation > {threshold}...")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print(f"Dropped {len(to_drop)} features.")
    
    return X_train.drop(columns=to_drop), X_val.drop(columns=to_drop), X_test.drop(columns=to_drop)

def scale_features(X_train, X_val, X_test):
    print("Applying Z-score normalization...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def apply_smote(X_train, y_train, random_state: int):
    print("Applying SMOTE...")
    smote = SMOTE(random_state=random_state)
    
    # Capture the output as a single tuple to appease the linter
    resampled = smote.fit_resample(X_train, y_train)
    
    # Explicitly return the first two items
    return resampled[0], resampled[1]

def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, save_dir):
    print(f"Saving processed datasets to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as parquet (requires pyarrow or fastparquet installed)
    X_train.to_parquet(save_dir / "X_train_smote.parquet")
    pd.DataFrame(y_train).to_parquet(save_dir / "y_train_smote.parquet")
    X_val.to_parquet(save_dir / "X_val.parquet")
    pd.DataFrame(y_val).to_parquet(save_dir / "y_val.parquet")
    X_test.to_parquet(save_dir / "X_test.parquet")
    pd.DataFrame(y_test).to_parquet(save_dir / "y_test.parquet")

def load_processed_data(save_dir):
    print(f"Loading processed datasets from {save_dir}...")
    X_train = pd.read_parquet(save_dir / "X_train_smote.parquet")
    y_train = pd.read_parquet(save_dir / "y_train_smote.parquet")['Label']
    X_val = pd.read_parquet(save_dir / "X_val.parquet")
    y_val = pd.read_parquet(save_dir / "y_val.parquet")['Label']
    X_test = pd.read_parquet(save_dir / "X_test.parquet")
    y_test = pd.read_parquet(save_dir / "y_test.parquet")['Label']
    
    return X_train, y_train, X_val, y_val, X_test, y_test
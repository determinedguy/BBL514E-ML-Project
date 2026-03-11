# src/preprocessing.py
"""
Data preprocessing and feature engineering
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .config import RANDOM_STATE, TEST_SIZE, FEATURES_TO_DROP

def clean_data(df):
    """Clean the dataset"""
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Remove duplicates
    print("\n1. Removing duplicates...")
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"   Removed {duplicates} duplicate rows")
    
    # 2. Handle missing values
    print("\n2. Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Drop columns with >50% missing
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        print(f"   Dropping columns with >50% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Fill remaining missing values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    print(f"   Handled {missing_before} missing values")
    
    # 3. Handle infinity values
    print("\n3. Handling infinity values...")
    # Recalculate numeric_cols to be safe
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum().sum()
    if inf_counts > 0:
        print(f"   Found {inf_counts} infinity values")
        # Replace inf with very large numbers
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])
        print(f"   Replaced infinity values with ±1e10")
    else:
        print(f"   No infinity values found")
    
    # 4. Remove irrelevant features
    print("\n4. Removing irrelevant features...")
    existing_features_to_drop = [f for f in FEATURES_TO_DROP if f in df.columns]
    if existing_features_to_drop:
        df = df.drop(columns=existing_features_to_drop)
        print(f"   Dropped: {existing_features_to_drop}")
    
    # 5. Remove constant/near-constant features
    print("\n5. Removing constant features...")
    # FIXED: Recalculate numeric_cols AFTER dropping irrelevant features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    nunique = df[numeric_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"   Dropped {len(constant_cols)} constant columns")
    else:
        print(f"   No constant columns found")
    
    final_rows = len(df)
    print(f"\n✓ Cleaning complete: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed)")
    
    return df

def prepare_features(df):
    """Prepare features and labels"""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    df = df.copy()
    
    # Separate features and labels
    if 'Label' not in df.columns:
        raise ValueError("'Label' column not found in dataset")
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {y.nunique()}")
    
    # Ensure all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"\n⚠ Warning: Non-numeric columns found: {non_numeric}")
        print(f"  Dropping non-numeric columns...")
        X = X.select_dtypes(include=[np.number])
    
    print(f"\nFinal features shape: {X.shape}")
    
    return X, y

def encode_labels(y_train, y_test):
    """Encode string labels to integers"""
    print("\n" + "="*60)
    print("LABEL ENCODING")
    print("="*60)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"\nOriginal labels: {le.classes_}")
    print(f"Encoded as: {list(range(len(le.classes_)))}")
    
    # Save to DATA_PROCESSED_DIR
    from .config import DATA_PROCESSED_DIR, LABEL_ENCODER_FILE
    le_path = os.path.join(DATA_PROCESSED_DIR, LABEL_ENCODER_FILE)
    joblib.dump(le, le_path)
    print(f"\n✓ Saved label encoder to: {le_path}")
    
    return y_train_encoded, y_test_encoded, le

def scale_features(X_train, X_test):
    """Scale features to zero mean and unit variance"""
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFeatures scaled using StandardScaler")
    print(f"Training set - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    print(f"Test set - Mean: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")
    
    # Save to DATA_PROCESSED_DIR
    from .config import DATA_PROCESSED_DIR, SCALER_FILE
    scaler_path = os.path.join(DATA_PROCESSED_DIR, SCALER_FILE)
    joblib.dump(scaler, scaler_path)
    print(f"\n✓ Saved scaler to: {scaler_path}")
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_data(df):
    """Complete preprocessing pipeline"""
    print("\n" + "="*60)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Clean data
    df_clean = clean_data(df)
    
    # 2. Prepare features and labels
    X, y = prepare_features(df_clean)
    
    # 3. Train-test split
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    print(f"\nSplit ratio: {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Stratified: Yes")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts())
    
    # 4. Encode labels
    y_train_encoded, y_test_encoded, le = encode_labels(y_train, y_test)
    
    # 5. Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 6. Save processed data
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    train_data = {
        'X': X_train_scaled,
        'y': y_train_encoded,
        'feature_names': X_train.columns.tolist()
    }
    test_data = {
        'X': X_test_scaled,
        'y': y_test_encoded,
        'feature_names': X_test.columns.tolist()
    }
    
    # Save to DATA_PROCESSED_DIR
    from .config import DATA_PROCESSED_DIR, PROCESSED_TRAIN, PROCESSED_TEST
    
    train_path = os.path.join(DATA_PROCESSED_DIR, PROCESSED_TRAIN)
    test_path = os.path.join(DATA_PROCESSED_DIR, PROCESSED_TEST)
    
    joblib.dump(train_data, train_path)
    joblib.dump(test_data, test_path)
    
    print(f"\n✓ Saved training data to: {train_path}")
    print(f"✓ Saved test data to: {test_path}")
    
    print("\n" + "="*60)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, le

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data(data_dir_path: Path) -> pd.DataFrame:
    print(f"Loading CSV files from {data_dir_path}...")
    
    # Use rglob to find all CSVs recursively, or glob for just that directory
    all_files = list(data_dir_path.glob("*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir_path}!")
    
    # Read and concatenate all CSVs
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Original shape: {df.shape}")
    
    # Standardize column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    print("Cleaning infinite values and NaNs...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print("Consolidating labels to binary classification with label encoding...")
    # Map BENIGN to 0, and all attacks to 1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Drop Identifiers / Metadata (Data Leakage Prevention)
    columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    print(f"Cleaned shape: {df.shape}")
    return df

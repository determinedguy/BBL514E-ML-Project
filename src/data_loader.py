# src/data_loader.py
"""
Load and explore CICIDS2017 dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import DATA_DIR, DATA_FILES, RESULTS_DIR

def load_single_file(filename):
    """Load a single CSV file"""
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Loading {filename}...")
    
    # Try different encodings
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"  Shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read {filename} with any encoding")

def load_all_data(sample_size=None):
    """Load all CICIDS2017 CSV files"""
    dfs = []
    
    for filename in DATA_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠ File not found: {filename}, skipping...")
            continue
        
        try:
            df = load_single_file(filename)
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data files were loaded successfully!")
    
    # Combine all dataframes
    print("\nCombining all files...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")
    
    # Clean column names (remove leading/trailing spaces)
    combined_df.columns = combined_df.columns.str.strip()
    
    # Sample if requested
    if sample_size and len(combined_df) > sample_size:
        print(f"\nSampling {sample_size} records...")
        combined_df = combined_df.sample(n=sample_size, random_state=42)
        print(f"Sampled shape: {combined_df.shape}")
    
    return combined_df

def explore_data(df):
    """Basic data exploration"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n--- Column Info ---")
    print(f"Total columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()[:10]}... (showing first 10)")
    
    if 'Label' in df.columns:
        print(f"\nTarget column: 'Label'")
    elif ' Label' in df.columns:
        print(f"\nTarget column: ' Label' (has leading space!)")
        df.rename(columns={' Label': 'Label'}, inplace=True)
    
    print("\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Total missing: {missing.sum()}")
        print("\nColumns with missing values:")
        print(missing[missing > 0].sort_values(ascending=False).head(10))
    else:
        print("✓ No missing values!")
    
    print("\n--- Label Distribution ---")
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        print(label_counts)
        print(f"\nTotal classes: {len(label_counts)}")
        print(f"Class balance ratio: {label_counts.min() / label_counts.max():.4f}")
    
    print("\n--- Infinity Values ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    total_inf = inf_counts.sum()
    print(f"Total infinity values: {total_inf}")
    if total_inf > 0:
        print("\nColumns with infinity values:")
        print(inf_counts[inf_counts > 0].sort_values(ascending=False).head(10))
    
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    print("\n--- Basic Statistics ---")
    print(df.describe().iloc[:, :5])  # Show first 5 numeric columns
    
    return df

def visualize_data(df):
    """Create visualizations"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if 'Label' not in df.columns:
        print("⚠ 'Label' column not found, skipping visualization")
        return
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Class distribution (bar chart)
    plt.figure(figsize=(14, 7))
    label_counts = df['Label'].value_counts()
    ax = label_counts.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Attack Type Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for i, v in enumerate(label_counts):
        ax.text(i, v + len(df)*0.01, f'{v:,}', ha='center', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {RESULTS_DIR}/class_distribution.png")
    plt.close()
    
    # 2. Class distribution (pie chart)
    plt.figure(figsize=(12, 12))
    colors = sns.color_palette('Set3', len(label_counts))
    label_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=colors, 
                      textprops={'fontsize': 10})
    plt.title('Attack Type Distribution (%)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_distribution_pie.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR}/class_distribution_pie.png")
    plt.close()
    
    # 3. Class imbalance visualization
    plt.figure(figsize=(10, 6))
    label_pct = (label_counts / len(df) * 100).sort_values(ascending=True)
    label_pct.plot(kind='barh', color='coral', edgecolor='black')
    plt.title('Class Imbalance (Percentage)', fontsize=14, fontweight='bold')
    plt.xlabel('Percentage (%)', fontsize=12)
    plt.ylabel('Attack Type', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'class_imbalance.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR}/class_imbalance.png")
    plt.close()

def load_and_explore(sample_size=None):
    """Main function to load and explore data"""
    print("="*60)
    print("STARTING DATA LOADING AND EXPLORATION")
    print("="*60)
    
    # Load data
    df = load_all_data(sample_size=sample_size)
    
    # Explore
    df = explore_data(df)
    
    # Visualize
    visualize_data(df)
    
    print("\n" + "="*60)
    print("✓ DATA EXPLORATION COMPLETE!")
    print("="*60)
    
    return df
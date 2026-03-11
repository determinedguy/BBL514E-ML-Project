# create_test_data.py
"""
Generate synthetic test data matching CICIDS2017 format
"""

import pandas as pd
import numpy as np

# Load one CICIDS2017 file to get column structure
sample_file = '/teamspace/lightning_storage/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv'
df = pd.read_csv(sample_file, nrows=100)

# Get feature columns (exclude Label)
feature_cols = [col for col in df.columns if col != 'Label']

# Generate synthetic data (random values within reasonable ranges)
n_samples = 1000

synthetic_data = {}
for col in feature_cols:
    if df[col].dtype in [np.float64, np.int64]:
        # Use min/max from original data
        min_val = df[col].min()
        max_val = df[col].max()
        synthetic_data[col] = np.random.uniform(min_val, max_val, n_samples)
    else:
        # For non-numeric, just repeat first value
        synthetic_data[col] = [df[col].iloc[0]] * n_samples

# Create dataframe
test_df = pd.DataFrame(synthetic_data)

# Save
test_df.to_csv('synthetic_test_data.csv', index=False)
print(f"✓ Created synthetic_test_data.csv with {n_samples} samples")
print(f"✓ Features: {len(feature_cols)}")

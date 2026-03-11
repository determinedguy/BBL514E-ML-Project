# extract_test_sample.py
"""
Extract a small sample from CICIDS2017 for testing
"""

import pandas as pd

# Load one file
file_path = '/teamspace/lightning_storage/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
df = pd.read_csv(file_path)

print(f"Original size: {len(df)} records")

# Sample different attack types
test_sample = df.groupby('Label').sample(n=50, random_state=42)

print(f"Test sample size: {len(test_sample)} records")
print("\nLabel distribution:")
print(test_sample['Label'].value_counts())

# Save WITHOUT labels (to simulate real prediction scenario)
test_features = test_sample.drop('Label', axis=1)
test_labels = test_sample[['Label']]

test_features.to_csv('test_input.csv', index=False)
test_labels.to_csv('test_ground_truth.csv', index=False)

print(f"\n✓ Created test_input.csv (features only)")
print(f"✓ Created test_ground_truth.csv (actual labels for comparison)")

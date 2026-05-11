from pathlib import Path

# Path Configurations
RAW_DATA_DIR = Path("/teamspace/lightning_storage/MachineLearningCVE") # Primary dataset location in Lightning AI
PROCESSED_DATA_DIR = Path("./data/processed")
MODEL_SAVE_DIR = Path("./models")
PLOT_DIR = Path("./plots")

# ML Hyperparameters
RANDOM_STATE = 42

# 70-15-15 split
TEMP_SPLIT_SIZE = 0.30       # 30% held out from train for Val+Test
VAL_TEST_SPLIT_SIZE = 0.50   # Splits that 30% equally into 15% Val / 15% Test

# Feature Selection
CORR_THRESHOLD = 0.95

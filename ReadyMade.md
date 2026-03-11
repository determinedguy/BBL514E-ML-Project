# Network Traffic Data Analyzer

A machine learning-based system for analyzing and classifying network traffic patterns using the CICIDS2017 dataset. This project implements multiple pattern recognition algorithms to detect various types of network attacks.

**Pattern Recognition & Analysis Course Project**

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Exploration](#1-data-exploration)
  - [Data Preprocessing](#2-data-preprocessing)
  - [Model Training](#3-model-training)
  - [Model Evaluation](#4-model-evaluation)
  - [Prediction](#5-prediction)
  - [Full Pipeline](#6-full-pipeline)
- [Models](#models)
- [Dataset](#dataset)
- [Output Files](#output-files)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Multiple ML Models**: Random Forest, MLP, SVM, KNN, and Ensemble
- **Complete Pipeline**: Data exploration → Preprocessing → Training → Evaluation → Prediction
- **Real-time Prediction**: Classify new network traffic from CSV files
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices
- **Visualization**: Class distribution charts, performance comparisons, confusion matrices
- **Flexible Sampling**: Work with sample or full dataset

---

## 📁 Project Structure

```
network-intrusion-detection/
├── data_processed/          # Processed training/test data
│   ├── train_data.pkl
│   ├── test_data.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── models/                  # Trained ML models
│   ├── random_forest.pkl
│   ├── mlp.pkl
│   ├── svm.pkl
│   ├── knn.pkl
│   └── ensemble.pkl
├── results/                 # Visualizations and metrics
│   ├── class_distribution.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── model_comparison.csv
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration
│   ├── data_loader.py      # Data loading & exploration
│   ├── preprocessing.py    # Data preprocessing
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
│   └── predict.py         # Prediction module
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the project**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- imbalanced-learn

3. **Prepare dataset**

Place CICIDS2017 CSV files in `/teamspace/lightning_storage/MachineLearningCVE/` or update the path in `src/config.py`.

---

## 🚀 Quick Start

**Run the complete pipeline:**

```bash
# Train and evaluate models with 500k samples
python main.py --mode full --sample 500000
```

This will:
1. ✅ Explore the data
2. ✅ Preprocess and split into train/test sets
3. ✅ Train all 5 models
4. ✅ Evaluate and compare models
5. ✅ Generate visualizations

**Results will be saved in:**
- `models/` - Trained models
- `results/` - Performance metrics and charts
- `data_processed/` - Preprocessed data

---

## 📖 Usage

### 1. Data Exploration

Analyze the dataset and generate visualizations:

```bash
# Explore with 100k samples
python main.py --mode explore --sample 100000

# Explore full dataset
python main.py --mode explore --no-sample
```

**Outputs:**
- Class distribution charts
- Data quality statistics
- Feature information

### 2. Data Preprocessing

Clean and prepare data for training:

```bash
# Preprocess with 500k samples
python main.py --mode preprocess --sample 500000
```

**This performs:**
- Duplicate removal
- Missing value handling
- Infinity value replacement
- Feature scaling (StandardScaler)
- Label encoding
- Train-test split (80/20)

**Outputs:**
- `data_processed/train_data.pkl`
- `data_processed/test_data.pkl`
- `data_processed/scaler.pkl`
- `data_processed/label_encoder.pkl`

### 3. Model Training

Train all machine learning models:

```bash
python main.py --mode train
```

**Models trained:**
1. **Random Forest** - Ensemble of 100 decision trees
2. **MLP** - Neural network with 2 hidden layers (128, 64 neurons)
3. **SVM** - Support Vector Machine with RBF kernel (trained on 30% sample)
4. **KNN** - K-Nearest Neighbors (k=5, trained on 20% sample)
5. **Ensemble** - Voting classifier combining RF + MLP

**Training time:** ~5-20 minutes depending on sample size

**Outputs:**
- `models/random_forest.pkl`
- `models/mlp.pkl`
- `models/svm.pkl`
- `models/knn.pkl`
- `models/ensemble.pkl`

### 4. Model Evaluation

Evaluate and compare all trained models:

```bash
python main.py --mode evaluate
```

**Metrics calculated:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Prediction Time

**Outputs:**
- `results/model_comparison.png` - Performance comparison chart
- `results/model_comparison.csv` - Detailed metrics table
- `results/confusion_matrices.png` - Confusion matrices for all models
- `results/prediction_time.png` - Prediction time comparison

### 5. Prediction

Classify new network traffic:

```bash
# Basic prediction (display results)
python main.py --mode predict --input test_data.csv --model ensemble

# Save predictions to file
python main.py --mode predict --input test_data.csv --model ensemble --output predictions.csv

# Use specific model
python main.py --mode predict --input test_data.csv --model rf --output results.csv
```

**Available models:**
- `rf` - Random Forest
- `mlp` - Neural Network
- `svm` - Support Vector Machine
- `knn` - K-Nearest Neighbors
- `ensemble` - Ensemble (default, recommended)

**Input CSV format:**
- Must contain the same features as CICIDS2017
- Can include or exclude 'Label' column (will be ignored if present)

**Output format:**
```csv
Prediction,Prediction_Code,Confidence
BENIGN,0,0.98
DDoS,1,0.95
PortScan,5,0.87
...
```

### 6. Full Pipeline

Run everything in sequence:

```bash
# Full pipeline with sampling
python main.py --mode full --sample 500000

# Full pipeline with complete dataset
python main.py --mode full --no-sample
```

---

## 🤖 Models

### Random Forest
- **Parameters:** 100 trees, max depth 20
- **Pros:** Fast, accurate, robust to overfitting
- **Training time:** ~2-5 minutes

### MLP (Neural Network)
- **Architecture:** Input → 128 → 64 → Output
- **Activation:** ReLU (hidden), Softmax (output)
- **Pros:** Captures non-linear patterns
- **Training time:** ~5-10 minutes

### SVM with RBF Kernel
- **Kernel:** Radial Basis Function
- **Note:** Trained on 30% sample for speed
- **Pros:** Strong theoretical foundation
- **Training time:** ~3-8 minutes

### KNN
- **Parameters:** k=5 neighbors
- **Note:** Trained on 20% sample for efficiency
- **Pros:** Simple, interpretable
- **Training time:** ~1-2 minutes

### Ensemble (Recommended)
- **Combination:** Random Forest + MLP (soft voting)
- **Pros:** Best overall performance
- **Training time:** Instant (combines existing models)

---

## 📊 Dataset

**CICIDS2017** - Canadian Institute for Cybersecurity Intrusion Detection System

**Attack Types:**
- BENIGN - Normal traffic
- DDoS - Distributed Denial of Service
- DoS - Denial of Service (Hulk, Slowloris, GoldenEye)
- PortScan - Network reconnaissance
- Brute Force - SSH/FTP attacks
- Web Attack - SQL Injection, XSS
- Botnet - ARES botnet traffic
- Infiltration - Network infiltration attacks

**Features:** 83 flow-based features including:
- Flow duration, packets, bytes
- Forward/backward statistics
- Inter-Arrival Time (IAT)
- Flag counts
- Header information

**Dataset location:** `/teamspace/lightning_storage/MachineLearningCVE/`

---

## 📂 Output Files

### Models Directory (`models/`)
```
random_forest.pkl    # Random Forest classifier
mlp.pkl             # Neural network
svm.pkl             # Support Vector Machine
knn.pkl             # K-Nearest Neighbors
ensemble.pkl        # Ensemble model
```

### Results Directory (`results/`)
```
class_distribution.png       # Attack type distribution (bar chart)
class_distribution_pie.png   # Attack type distribution (pie chart)
class_imbalance.png         # Class imbalance visualization
model_comparison.png        # Model performance comparison
model_comparison.csv        # Detailed metrics table
confusion_matrices.png      # Confusion matrices for all models
prediction_time.png         # Prediction time comparison
```

### Processed Data Directory (`data_processed/`)
```
train_data.pkl         # Training features and labels
test_data.pkl          # Test features and labels
scaler.pkl            # StandardScaler for feature normalization
label_encoder.pkl     # Label encoder for attack type mapping
```

---

## 💡 Examples

### Example 1: Quick Test with Small Sample

```bash
# Explore 50k samples
python main.py --mode explore --sample 50000

# Train on 50k samples
python main.py --mode full --sample 50000

# Predict on new data
python main.py --mode predict --input test.csv --model ensemble
```

### Example 2: Full Production Pipeline

```bash
# Step 1: Explore full dataset
python main.py --mode explore --no-sample

# Step 2: Preprocess
python main.py --mode preprocess --no-sample

# Step 3: Train models
python main.py --mode train

# Step 4: Evaluate
python main.py --mode evaluate

# Step 5: Use best model for predictions
python main.py --mode predict --input new_traffic.csv --model ensemble --output predictions.csv
```

### Example 3: Model Comparison

```bash
# Compare different models
python main.py --mode predict --input test.csv --model rf --output rf_pred.csv
python main.py --mode predict --input test.csv --model mlp --output mlp_pred.csv
python main.py --mode predict --input test.csv --model ensemble --output ensemble_pred.csv

# View evaluation metrics
cat results/model_comparison.csv
```

### Example 4: Create Test Data from Training Set

```python
# extract_test_sample.py
import pandas as pd

# Load a CICIDS file
df = pd.read_csv('/teamspace/lightning_storage/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')

# Sample 50 records per attack type
test_sample = df.groupby('Label').sample(n=50, random_state=42)

# Save features only (for prediction)
test_features = test_sample.drop('Label', axis=1)
test_features.to_csv('test_input.csv', index=False)

# Save labels separately (for validation)
test_sample[['Label']].to_csv('test_ground_truth.csv', index=False)

print(f"Created test_input.csv with {len(test_features)} samples")
```

Then predict:
```bash
python main.py --mode predict --input test_input.csv --model ensemble --output predictions.csv
```

---

## 🐛 Troubleshooting

### Issue: "No CSV files found"
**Solution:** Update `DATA_DIR` in `src/config.py` to point to your CICIDS2017 location.

### Issue: "Processed data not found"
**Solution:** Run preprocessing first:
```bash
python main.py --mode preprocess --sample 500000
```

### Issue: "Model not found"
**Solution:** Train models first:
```bash
python main.py --mode train
```

### Issue: Out of memory
**Solution:** Use smaller sample size:
```bash
python main.py --mode full --sample 100000
```

### Issue: Slow training
**Solution:** 
- Use smaller sample for initial testing
- SVM and KNN are already trained on reduced data
- Consider using only RF and MLP for faster iteration

### Issue: Import errors
**Solution:** Ensure you're running from project root:
```bash
cd /path/to/network-intrusion-detection
python main.py --mode explore
```

---

## 📈 Performance Expectations

**Expected Accuracy (on CICIDS2017):**
- Random Forest: ~95-98%
- MLP: ~94-97%
- SVM: ~92-95%
- KNN: ~88-92%
- Ensemble: ~96-99%

**Training Time (500k samples):**
- Random Forest: ~3-5 minutes
- MLP: ~5-8 minutes
- SVM: ~4-6 minutes (30% sample)
- KNN: ~1-2 minutes (20% sample)
- Ensemble: ~1 second (combines existing)

**Prediction Time (1000 samples):**
- All models: <1 second

---

## 🎯 Project Goals

This project demonstrates:
1. ✅ Data exploration and preprocessing
2. ✅ Implementation of multiple pattern recognition algorithms
3. ✅ Performance comparison and evaluation
4. ✅ Practical deployment for real-world predictions
5. ✅ Comprehensive documentation and visualization

---

## 📝 Notes

- **Sampling:** Use `--sample` for faster iteration during development. Use `--no-sample` for final results.
- **Models:** Ensemble model typically provides best results. Use individual models for comparison.
- **Dataset:** System auto-detects all CSV files in data directory.
- **Memory:** Adjust sample size based on available RAM (500k samples ≈ 2-4GB).

---

## 🤝 Contributing

This is a course project. For questions or improvements, please contact the project team.

---

## 📄 License

Educational use only - Pattern Recognition & Analysis Course Project

---

**Project Status:** ✅ Fully Functional

**Last Updated:** March 2026

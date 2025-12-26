"""
Configuration settings for the Digit Recognition Application.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = BASE_DIR
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Model storage
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# Processed data storage
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
PROCESSED_X_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train.npy")
PROCESSED_Y_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train.npy")
PROCESSED_X_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test.npy")

# Model hyperparameters
N_CLUSTERS = 50
PCA_VARIANCE_RATIO = 0.95
MAX_ITER_KMEANS = 300
MAX_ITER_LOGREG = 5000
RANDOM_STATE = 42
KMEANS_TOL = 1e-4

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

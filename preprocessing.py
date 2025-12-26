"""
Data preprocessing module for Digit Recognition.
"""
import os
import numpy as np
import pandas as pd
from config import (
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    PROCESSED_X_TRAIN_PATH, PROCESSED_Y_TRAIN_PATH, PROCESSED_X_TEST_PATH
)


def load_raw_data():
    """
    Load raw training and test data from CSV files.
    
    Returns
    -------
    tuple
        (train_df, test_df) DataFrames
    """
    print("📂 Loading raw data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"   Train data shape: {train_df.shape}")
    print(f"   Test data shape: {test_df.shape}")
    return train_df, test_df


def preprocess_data(train_df, test_df=None):
    """
    Preprocess data by separating features and labels.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with 'label' column.
    test_df : pd.DataFrame, optional
        Test data (no label column).
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test) numpy arrays
    """
    print("🔧 Preprocessing data...")
    
    # Extract features and labels from training data
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    
    # Process test data if provided
    X_test = test_df.values if test_df is not None else None
    
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    if X_test is not None:
        print(f"   X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test


def save_processed_data(X_train, y_train, X_test=None):
    """
    Save processed data arrays to disk.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray, optional
        Test features.
    """
    print("💾 Saving processed data...")
    np.save(PROCESSED_X_TRAIN_PATH, X_train)
    np.save(PROCESSED_Y_TRAIN_PATH, y_train)
    if X_test is not None:
        np.save(PROCESSED_X_TEST_PATH, X_test)
    print("   ✅ Processed data saved successfully!")


def load_processed_data():
    """
    Load processed data arrays from disk.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test) numpy arrays, or None if not found
    """
    if not processed_data_exists():
        return None, None, None
    
    print("📂 Loading processed data from cache...")
    X_train = np.load(PROCESSED_X_TRAIN_PATH)
    y_train = np.load(PROCESSED_Y_TRAIN_PATH)
    X_test = None
    if os.path.exists(PROCESSED_X_TEST_PATH):
        X_test = np.load(PROCESSED_X_TEST_PATH)
    
    print(f"   ✅ Loaded from cache!")
    return X_train, y_train, X_test


def processed_data_exists():
    """Check if processed data files exist."""
    return (os.path.exists(PROCESSED_X_TRAIN_PATH) and 
            os.path.exists(PROCESSED_Y_TRAIN_PATH))


def get_data(use_cache=True):
    """
    Get preprocessed data, using cache if available.
    
    Parameters
    ----------
    use_cache : bool
        Whether to use cached processed data if available.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test) numpy arrays
    """
    if use_cache and processed_data_exists():
        return load_processed_data()
    
    train_df, test_df = load_raw_data()
    X_train, y_train, X_test = preprocess_data(train_df, test_df)
    save_processed_data(X_train, y_train, X_test)
    
    return X_train, y_train, X_test

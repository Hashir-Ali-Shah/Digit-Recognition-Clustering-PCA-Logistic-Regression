"""
Model persistence manager for Digit Recognition.
"""
import os
import joblib
from config import PIPELINE_PATH, MODELS_DIR


def model_exists():
    """
    Check if a trained model exists on disk.
    
    Returns
    -------
    bool
        True if model file exists
    """
    return os.path.exists(PIPELINE_PATH)


def save_model(pipeline):
    """
    Save trained pipeline to disk.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline to save.
    """
    print("💾 Saving model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"   ✅ Model saved to: {PIPELINE_PATH}")


def load_model():
    """
    Load trained pipeline from disk.
    
    Returns
    -------
    sklearn.pipeline.Pipeline or None
        Loaded pipeline, or None if not found
    """
    if not model_exists():
        print("⚠️  No saved model found.")
        return None
    
    print("📂 Loading saved model...")
    pipeline = joblib.load(PIPELINE_PATH)
    print("   ✅ Model loaded successfully!")
    return pipeline


def get_or_train_model(X_train=None, y_train=None, force_retrain=False):
    """
    Get existing model or train a new one.
    
    Parameters
    ----------
    X_train : np.ndarray, optional
        Training features (required if training new model).
    y_train : np.ndarray, optional
        Training labels (required if training new model).
    force_retrain : bool
        Force retraining even if model exists.
    
    Returns
    -------
    tuple
        (pipeline, is_newly_trained)
    """
    from training import build_pipeline, train_model
    
    if not force_retrain and model_exists():
        pipeline = load_model()
        return pipeline, False
    
    if X_train is None or y_train is None:
        raise ValueError("Training data required to train a new model")
    
    pipeline = build_pipeline()
    pipeline = train_model(pipeline, X_train, y_train)
    save_model(pipeline)
    
    return pipeline, True

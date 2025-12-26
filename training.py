"""
Model training pipeline for Digit Recognition.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    N_CLUSTERS, PCA_VARIANCE_RATIO, MAX_ITER_KMEANS, 
    MAX_ITER_LOGREG, RANDOM_STATE, KMEANS_TOL
)
from models import CustomKMeans


def build_pipeline():
    """
    Build the complete ML pipeline.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Complete pipeline with Scaler, PCA, KMeans, and LogisticRegression
    """
    print("🔨 Building ML pipeline...")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_STATE)),
        ('kmeans', CustomKMeans(
            n_clusters=N_CLUSTERS, 
            max_iter=MAX_ITER_KMEANS,
            tol=KMEANS_TOL,
            random_state=RANDOM_STATE
        )),
        ('log_reg', LogisticRegression(
            max_iter=MAX_ITER_LOGREG, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    print("   ✅ Pipeline built successfully!")
    return pipeline


def train_model(pipeline, X_train, y_train):
    """
    Train the pipeline on training data.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The ML pipeline to train.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained pipeline
    """
    print("=" * 60)
    print("🚀 STARTING MODEL TRAINING")
    print("=" * 60)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Unique labels: {len(np.unique(y_train))}")
    print("-" * 60)
    
    print("[Step 1/4] Fitting StandardScaler...")
    X_scaled = pipeline.named_steps['scaler'].fit_transform(X_train)
    print(f"   ✅ Scaler complete. Shape: {X_scaled.shape}")
    
    print("[Step 2/4] Fitting PCA for dimensionality reduction...")
    X_pca = pipeline.named_steps['pca'].fit_transform(X_scaled)
    n_components = pipeline.named_steps['pca'].n_components_
    print(f"   ✅ PCA complete. Reduced to {n_components} components")
    
    print("[Step 3/4] Fitting Custom KMeans clustering...")
    X_kmeans = pipeline.named_steps['kmeans'].fit_transform(X_pca)
    print(f"   ✅ KMeans complete. Distance features shape: {X_kmeans.shape}")
    
    print("[Step 4/4] Fitting Logistic Regression classifier...")
    pipeline.named_steps['log_reg'].fit(X_kmeans, y_train)
    print("   ✅ Logistic Regression complete.")
    
    print("-" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60)
    return pipeline


def evaluate_model(pipeline, X, y):
    """
    Evaluate the trained pipeline.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline.
    X : np.ndarray
        Features to evaluate on.
    y : np.ndarray
        True labels.
    
    Returns
    -------
    dict
        Evaluation metrics including accuracy, classification report, confusion matrix
    """
    print("📊 Evaluating model...")
    
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    
    print(f"   Accuracy: {accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }


def get_cluster_distribution(pipeline, X, y):
    """
    Analyze cluster distribution of actual labels.
    
    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline.
    X : np.ndarray
        Features.
    y : np.ndarray
        True labels.
    
    Returns
    -------
    pd.DataFrame
        Cluster-label distribution
    """
    import pandas as pd
    
    # Get cluster labels from the trained KMeans
    kmeans = pipeline.named_steps['kmeans']
    cluster_labels = kmeans.labels_
    
    result = pd.DataFrame({
        'Actual_Label': y, 
        'Cluster_Label': cluster_labels
    })
    
    distribution = result.groupby(['Cluster_Label', 'Actual_Label']).size()
    return distribution

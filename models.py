"""
Custom ML model implementations for Digit Recognition.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist


class CustomKMeans(BaseEstimator, TransformerMixin):
    """
    Custom KMeans clustering implementation compatible with sklearn Pipeline.
    Uses memory-efficient distance computation.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    random_state : int or None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Fit the KMeans model on data X."""
        print(f"[CustomKMeans] Starting fit with {len(X)} samples, {self.n_clusters} clusters...")
        X = np.array(X, dtype=np.float32)  # Use float32 to reduce memory
        rng = np.random.default_rng(self.random_state)
        
        # Initialize cluster centers randomly
        random_idx = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_idx].copy()
        print(f"[CustomKMeans] Initialized {self.n_clusters} cluster centers")

        for iteration in range(self.max_iter):
            # Assign points to nearest cluster using memory-efficient cdist
            distances = cdist(X, self.cluster_centers_, metric='euclidean')
            labels = np.argmin(distances, axis=1)

            # Update cluster centers
            new_centers = np.zeros_like(self.cluster_centers_)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # Check for convergence
            shift = np.linalg.norm(new_centers - self.cluster_centers_)
            self.cluster_centers_ = new_centers
            
            # Print progress every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"[CustomKMeans] Iteration {iteration + 1}/{self.max_iter}, shift: {shift:.6f}")
            
            if shift < self.tol:
                print(f"[CustomKMeans] Converged at iteration {iteration + 1}")
                break

        self.labels_ = labels
        print(f"[CustomKMeans] Fit complete!")
        return self

    def transform(self, X):
        """Transform X to cluster-distance space."""
        X = np.array(X, dtype=np.float32)
        return cdist(X, self.cluster_centers_, metric='euclidean')

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Predict the closest cluster for each sample in X."""
        X = np.array(X, dtype=np.float32)
        distances = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(distances, axis=1)

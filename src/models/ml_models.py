"""
Traditional ML models: Random Forest and SVM.

Uses scikit-learn with GridSearchCV for hyperparameter tuning.
"""

import pickle
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class MLModelTrainer:
    """Trainer for traditional ML models (Random Forest, SVM)."""

    def __init__(self, model_type: str = 'rf'):
        """
        Initialize ML model trainer.

        Args:
            model_type: 'rf' for Random Forest, 'svm' for SVM
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

        if model_type == 'rf':
            self.param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': [None, 'balanced'],
            }
            self.base_model = RandomForestClassifier(random_state=42)
        elif model_type == 'svm':
            self.param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
            }
            self.base_model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'rf' or 'svm'.")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train the model with GridSearchCV.

        Args:
            X_train: Training features, shape (n_samples, n_features)
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            metrics: Dict with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Grid search
        grid_search = GridSearchCV(
            self.base_model,
            self.param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train_scaled, y_train)

        self.model = grid_search.best_estimator_
        self.is_fitted = True

        # Training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_proba = self.model.predict_proba(X_train_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None

        metrics = {
            'model_type': self.model_type,
            'best_params': grid_search.best_params_,
            'best_cv_score': float(grid_search.best_score_),
            'train_accuracy': float(np.mean(train_pred == y_train)),
        }

        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            metrics['val_accuracy'] = float(np.mean(val_pred == y_val))

            if train_proba is not None:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                val_proba = self.model.predict_proba(X_val_scaled)[:, 1]
                metrics['val_precision'] = float(precision_score(y_val, val_pred, zero_division=0))
                metrics['val_recall'] = float(recall_score(y_val, val_pred, zero_division=0))
                metrics['val_f1'] = float(f1_score(y_val, val_pred, zero_division=0))
                try:
                    metrics['val_roc_auc'] = float(roc_auc_score(y_val, val_proba))
                except ValueError:
                    metrics['val_roc_auc'] = 0.0

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            predictions: numpy array of class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features, shape (n_samples, n_features)

        Returns:
            probabilities: numpy array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance or coefficients.

        Args:
            feature_names: List of feature names

        Returns:
            importance: Dict mapping feature names to importance values
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        elif self.model_type == 'svm':
            # For SVM, use absolute coefficients
            if hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_).mean(axis=0)
            else:
                return {}
        else:
            return {}

        return dict(zip(feature_names, importances.tolist()))

    def save(self, path: str) -> None:
        """Save model and scaler to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        """Load model and scaler from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data['model_type']
        self.is_fitted = True

"""
Traditional ML models: Random Forest and SVM.
Uses scikit-learn with GridSearchCV for hyperparameter tuning.
"""
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class MLModelTrainer:
    """Trainer for traditional ML models (Random Forest, SVM)."""

    def __init__(self, model_type: str = 'rf', feature_selection: bool = True):
        self.model_type = model_type
        self.feature_selection = feature_selection
        self.model = None
        self.scaler = StandardScaler()
        self.selector = None
        self.is_fitted = False
        self.selected_features_mask = None

        if model_type == 'rf':
            self.param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced'],
            }
            self.base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            self.param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced'],
            }
            self.base_model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'rf' or 'svm'.")

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Apply RFECV for feature selection."""
        if not self.feature_selection or X.shape[1] <= 10:
            self.selected_features_mask = np.ones(X.shape[1], dtype=bool)
            return X

        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) if self.model_type == 'rf' else SVC(kernel='linear', C=1, random_state=42)
        cv = StratifiedKFold(2, shuffle=True, random_state=42)
        self.selector = RFECV(estimator, step=0.15, cv=cv, scoring='f1', n_jobs=-1, min_features_to_select=20)
        self.selector.fit(X, y)
        self.selected_features_mask = self.selector.support_
        n_selected = int(self.selected_features_mask.sum())
        print(f"  RFECV: {X.shape[1]} -> {n_selected} features selected")
        return self.selector.transform(X)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict:
        """Train the model with GridSearchCV and optional feature selection."""
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Feature selection on training data
        X_train_selected = self._select_features(X_train_scaled, y_train)

        n_iter = 20 if self.model_type == 'rf' else 12
        random_search = RandomizedSearchCV(
            self.base_model, self.param_grid, n_iter=n_iter, cv=2,
            scoring='f1', n_jobs=-1, verbose=0, random_state=42,
        )
        random_search.fit(X_train_selected, y_train)

        self.model = random_search.best_estimator_
        self.is_fitted = True

        train_pred = self.model.predict(X_train_selected)
        metrics = {
            'model_type': self.model_type,
            'best_params': random_search.best_params_,
            'best_cv_score': float(random_search.best_score_),
            'train_accuracy': float(np.mean(train_pred == y_train)),
            'n_features': X_train.shape[1],
            'n_features_selected': int(self.selected_features_mask.sum()) if self.selected_features_mask is not None else X_train.shape[1],
        }

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_selected = X_val_scaled[:, self.selected_features_mask] if self.selected_features_mask is not None else X_val_scaled
            val_pred = self.model.predict(X_val_selected)
            metrics['val_accuracy'] = float(np.mean(val_pred == y_val))

            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            val_proba = self.model.predict_proba(X_val_selected)[:, 1]
            metrics['val_precision'] = float(precision_score(y_val, val_pred, zero_division=0))
            metrics['val_recall'] = float(recall_score(y_val, val_pred, zero_division=0))
            metrics['val_f1'] = float(f1_score(y_val, val_pred, zero_division=0))
            try:
                metrics['val_roc_auc'] = float(roc_auc_score(y_val, val_proba))
            except ValueError:
                metrics['val_roc_auc'] = 0.0

        print(f"  {self.model_type.upper()} best params: {random_search.best_params_}")
        print(f"  {self.model_type.upper()} CV F1: {random_search.best_score_:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features_mask] if self.selected_features_mask is not None else X_scaled
        return self.model.predict(X_selected)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")
        X_scaled = self.scaler.transform(X)
        X_selected = X_scaled[:, self.selected_features_mask] if self.selected_features_mask is not None else X_scaled
        return self.model.predict_proba(X_selected)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        if self.model_type == 'rf':
            importances = self.model.feature_importances_
        elif self.model_type == 'svm':
            if hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_).mean(axis=0)
            else:
                return {}
        else:
            return {}

        # Map back to original feature names via selected mask
        if self.selected_features_mask is not None:
            full_importances = np.zeros(len(feature_names))
            full_importances[self.selected_features_mask] = importances
            return {name: float(imp) for name, imp in zip(feature_names, full_importances) if imp > 0}
        return dict(zip(feature_names, importances.tolist()))

    def save(self, path: str) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'selected_features_mask': self.selected_features_mask,
            'model_type': self.model_type,
            'feature_selection': self.feature_selection,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.selector = data.get('selector')
        self.selected_features_mask = data.get('selected_features_mask')
        self.model_type = data['model_type']
        self.feature_selection = data.get('feature_selection', True)
        self.is_fitted = True

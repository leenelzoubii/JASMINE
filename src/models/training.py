"""
Training pipeline: cross-validation, metrics, and model comparison.

Runs all 4 models (RF, SVM, LSTM, Transformer) and produces comparison results.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)

    Returns:
        metrics: Dict of metric name -> value
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['roc_auc'] = 0.0

    return metrics


def run_ml_cv(X: np.ndarray, y: np.ndarray, feature_names: List[str],
              model_type: str = 'rf', cv_folds: int = 5) -> Dict:
    """
    Run cross-validation for an ML model.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        feature_names: List of feature names
        model_type: 'rf' or 'svm'
        cv_folds: Number of CV folds

    Returns:
        result: Dict with CV metrics and trained model
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        trainer = MLModelTrainer(model_type=model_type)
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        y_pred = trainer.predict(X_val)
        y_proba = trainer.predict_proba(X_val)[:, 1]

        fold_result = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(fold_result)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        print(f"  Fold {fold_idx+1}/{cv_folds} - {model_type.upper()} - "
              f"Acc: {fold_result['accuracy']:.4f} | F1: {fold_result['f1']:.4f}")

    # Aggregate metrics
    agg_metrics = compute_metrics(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)
    )
    agg_metrics['model_type'] = model_type
    agg_metrics['fold_metrics'] = fold_metrics

    # Train final model on all data
    final_trainer = MLModelTrainer(model_type=model_type)
    final_trainer.train(X, y)

    # Feature importance
    importance = final_trainer.get_feature_importance(feature_names)
    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
    agg_metrics['top_features'] = top_features

    return {
        'metrics': agg_metrics,
        'trainer': final_trainer,
    }


def run_dl_cv(X_sequences: List[np.ndarray], y: np.ndarray,
              model_type: str = 'lstm', cv_folds: int = 5,
              epochs: int = 30, batch_size: int = 32) -> Dict:
    """
    Run cross-validation for a DL model.

    Args:
        X_sequences: List of sequences, each shape (seq_len, n_features)
        y: Labels, shape (n_samples,)
        model_type: 'lstm' or 'transformer'
        cv_folds: Number of CV folds
        epochs: Training epochs per fold
        batch_size: Batch size

    Returns:
        result: Dict with CV metrics and trained model
    """
    input_size = X_sequences[0].shape[1] if len(X_sequences[0].shape) > 1 else 1

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_sequences, y)):
        X_train = [X_sequences[i] for i in train_idx]
        X_val = [X_sequences[i] for i in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        trainer = DLModelTrainer(model_type=model_type, input_size=input_size)
        metrics = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=0.001
        )

        y_pred = trainer.predict(X_val)
        y_proba = trainer.predict_proba(X_val)[:, 1]

        fold_result = compute_metrics(y_val, y_pred, y_proba)
        fold_metrics.append(fold_result)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        print(f"  Fold {fold_idx+1}/{cv_folds} - {model_type.upper()} - "
              f"Acc: {fold_result['accuracy']:.4f} | F1: {fold_result['f1']:.4f}")

    # Aggregate metrics
    agg_metrics = compute_metrics(
        np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)
    )
    agg_metrics['model_type'] = model_type
    agg_metrics['fold_metrics'] = fold_metrics

    # Train final model on all data
    final_trainer = DLModelTrainer(model_type=model_type, input_size=input_size)
    final_trainer.train(X_sequences, y, epochs=epochs, batch_size=batch_size)

    return {
        'metrics': agg_metrics,
        'trainer': final_trainer,
    }


def run_full_comparison(X: np.ndarray, y: np.ndarray, X_sequences: List[np.ndarray],
                        feature_names: List[str], cv_folds: int = 5,
                        dl_epochs: int = 30) -> Dict:
    """
    Run full comparison of all 4 models.

    Args:
        X: Feature matrix for ML models, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        X_sequences: Sequences for DL models, list of (seq_len, n_features)
        feature_names: List of feature names
        cv_folds: Number of CV folds
        dl_epochs: Training epochs for DL models

    Returns:
        results: Dict with all model results and comparison table
    """
    results = {}

    print("=" * 60)
    print("AUTISM SCREENING - MODEL COMPARISON")
    print("=" * 60)

    # ML Models
    for model_type in ['rf', 'svm']:
        print(f"\nTraining {model_type.upper()}...")
        result = run_ml_cv(X, y, feature_names, model_type, cv_folds)
        results[model_type] = result

    # DL Models
    for model_type in ['lstm', 'transformer']:
        print(f"\nTraining {model_type.upper()}...")
        result = run_dl_cv(X_sequences, y, model_type, cv_folds, epochs=dl_epochs)
        results[model_type] = result

    # Build comparison table
    comparison = []
    for model_type in ['rf', 'svm', 'lstm', 'transformer']:
        m = results[model_type]['metrics']
        comparison.append({
            'Model': model_type.upper(),
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1': f"{m['f1']:.4f}",
            'ROC-AUC': f"{m.get('roc_auc', 0):.4f}",
        })

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    print("-" * 75)
    for row in comparison:
        print(f"{row['Model']:<15} {row['Accuracy']:<12} {row['Precision']:<12} "
              f"{row['Recall']:<12} {row['F1']:<12} {row['ROC-AUC']:<12}")

    # Save results
    results_dict = {
        'comparison': comparison,
        'models': {},
    }

    for model_type in ['rf', 'svm', 'lstm', 'transformer']:
        m = results[model_type]['metrics']
        results_dict['models'][model_type] = {
            'accuracy': m['accuracy'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1': m['f1'],
            'roc_auc': m.get('roc_auc', 0),
            'confusion_matrix': m['confusion_matrix'],
            'top_features': m.get('top_features', {}),
        }

    # Return both comparison data AND trainer objects for saving
    results_dict['_trainers'] = {mt: results[mt]['trainer'] for mt in ['rf', 'svm', 'lstm', 'transformer']}
    return results_dict


def save_results(results: Dict, output_dir: str) -> str:
    """
    Save comparison results to JSON and models to disk.

    Args:
        results: Results dict from run_full_comparison
        output_dir: Directory to save results

    Returns:
        output_path: Path to saved JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save comparison JSON
    json_path = os.path.join(output_dir, 'comparison_results.json')

    # Convert non-serializable items
    saveable = {
        'comparison': results['comparison'],
        'models': results['models'],
    }

    with open(json_path, 'w') as f:
        json.dump(saveable, f, indent=2)

    print(f"\nResults saved to {json_path}")

    # Save individual models
    trainers = results.get('_trainers', {})
    for model_type in ['rf', 'svm', 'lstm', 'transformer']:
        if model_type in trainers:
            trainer = trainers[model_type]
            if model_type in ['rf', 'svm']:
                model_path = os.path.join(output_dir, f'{model_type}_model.pkl')
                trainer.save(model_path)
                print(f"Model saved to {model_path}")
            else:
                model_path = os.path.join(output_dir, f'{model_type}_model.pth')
                trainer.save(model_path)
                print(f"Model saved to {model_path}")

    return json_path

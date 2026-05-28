"""
Shared utilities for the Streamlit app.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent


def get_db_connection():
    DB_PATH = PROJECT_ROOT / "users.db"
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_all_models(models_dir: str) -> Dict:
    """
    Load all trained models from the models directory.

    Args:
        models_dir: Path to directory containing saved models

    Returns:
        models: Dict with model_type -> trainer instance
    """
    from src.models.ml_models import MLModelTrainer
    from src.models.dl_models import DLModelTrainer

    models = {}

    # Load ML models
    for model_type in ['rf', 'svm']:
        model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
        if os.path.exists(model_path):
            trainer = MLModelTrainer(model_type=model_type)
            trainer.load(model_path)
            models[model_type] = trainer

    # Load DL models
    for model_type in ['lstm', 'transformer']:
        model_path = os.path.join(models_dir, f'{model_type}_model.pth')
        if os.path.exists(model_path):
            trainer = DLModelTrainer(model_type=model_type)
            trainer.load(model_path)
            models[model_type] = trainer

    return models


def load_ensemble_weights(results_path: str) -> Dict[str, float]:
    """Load ensemble weights from comparison_results.json."""
    default_weights = {'rf': 0.34, 'svm': 0.28, 'lstm': 0.19, 'transformer': 0.19}
    try:
        if os.path.exists(results_path):
            import json
            with open(results_path, 'r') as f:
                data = json.load(f)
            return data.get('ensemble_weights', default_weights)
    except Exception:
        pass
    return default_weights


def get_ensemble_prediction(models: Dict, features: np.ndarray,
                            sequence: np.ndarray) -> Dict[str, float]:
    """
    Get predictions from all loaded models and compute ensemble.

    Args:
        models: Dict of loaded model trainers
        features: Feature vector for ML models, shape (n_features,)
        sequence: Keypoint sequence for DL models, shape (seq_len, n_features)

    Returns:
        predictions: Dict with model_type -> ASD probability
    """
    predictions = {}

    # ML model predictions
    for model_type in ['rf', 'svm']:
        if model_type in models:
            features_2d = features.reshape(1, -1)
            proba = models[model_type].predict_proba(features_2d)[0]
            predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0

    # DL model predictions
    for model_type in ['lstm', 'transformer']:
        if model_type in models:
            sequence_list = [sequence]
            proba = models[model_type].predict_proba(sequence_list)[0]
            predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0

    return predictions


def get_risk_level(asd_probability: float) -> Tuple[str, str]:
    """
    Determine risk level from ASD probability.

    Args:
        asd_probability: Probability of ASD classification

    Returns:
        risk_level: 'Low Risk', 'Moderate Risk', or 'High Risk'
        color: Color code for display
    """
    if asd_probability < 0.3:
        return 'Low Risk', '#2ca02c'
    elif asd_probability < 0.6:
        return 'Moderate Risk', '#ff7f0e'
    else:
        return 'High Risk', '#d62728'


def format_prediction_result(predictions: Dict[str, float],
                             weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Format prediction results for display using weighted ensemble.

    Args:
        predictions: Dict with model_type -> ASD probability
        weights: Optional dict of ensemble weights per model

    Returns:
        result: Formatted dict for display
    """
    if weights is None:
        weights = {'rf': 0.34, 'svm': 0.28, 'lstm': 0.19, 'transformer': 0.19}

    weighted_sum = 0.0
    total_weight = 0.0
    for model_type, prob in predictions.items():
        w = weights.get(model_type, 0.25)
        weighted_sum += prob * w
        total_weight += w

    ensemble_prob = weighted_sum / total_weight if total_weight > 0 else (
        np.mean(list(predictions.values())) if predictions else 0.0
    )
    risk_level, color = get_risk_level(ensemble_prob)

    result = {
        'ensemble_probability': float(ensemble_prob),
        'risk_level': risk_level,
        'risk_color': color,
        'model_predictions': {},
    }

    model_display_names = {
        'rf': 'Random Forest',
        'svm': 'SVM',
        'lstm': 'LSTM',
        'transformer': 'Transformer',
    }

    for model_type, prob in predictions.items():
        display_name = model_display_names.get(model_type, model_type)
        model_risk, model_color = get_risk_level(prob)
        result['model_predictions'][display_name] = {
            'probability': prob,
            'risk_level': model_risk,
            'color': model_color,
        }

    return result


def generate_report(subject_id: str, formatted_result: Dict,
                    feature_contributions: Optional[Dict] = None) -> str:
    """
    Generate a text prediction report.

    Args:
        subject_id: Subject identifier
        formatted_result: Formatted prediction result
        feature_contributions: Optional dict of feature contributions

    Returns:
        report: Text report string
    """
    lines = [
        "=" * 60,
        "AUTISM SCREENING PREDICTION REPORT",
        "=" * 60,
        f"Subject ID: {subject_id}",
        f"Date: Generated by Autism Screening Demo",
        "",
        "ENSEMBLE RESULT",
        "-" * 30,
        f"ASD Probability: {formatted_result['ensemble_probability']:.1%}",
        f"Risk Level: {formatted_result['risk_level']}",
        "",
        "INDIVIDUAL MODEL PREDICTIONS",
        "-" * 30,
    ]

    for model_name, model_data in formatted_result['model_predictions'].items():
        lines.append(
            f"  {model_name}: {model_data['probability']:.1%} ({model_data['risk_level']})"
        )

    if feature_contributions:
        lines.extend([
            "",
            "TOP CONTRIBUTING FEATURES",
            "-" * 30,
        ])
        for feat_name, contribution in list(feature_contributions.items())[:10]:
            lines.append(f"  {feat_name}: {contribution:.4f}")

    lines.extend([
        "",
        "DISCLAIMER",
        "-" * 30,
        "This is a research demo and NOT a diagnostic tool.",
        "Results should not be used for clinical decision-making.",
        "Consult a qualified healthcare professional for diagnosis.",
        "=" * 60,
    ])

    return "\n".join(lines)


def load_comparison_results(results_path: str) -> Optional[Dict]:
    """
    Load model comparison results from JSON file.

    Args:
        results_path: Path to comparison_results.json

    Returns:
        results: Dict with comparison data, or None if not found
    """
    import json

    if not os.path.exists(results_path):
        return None

    with open(results_path, 'r') as f:
        return json.load(f)

"""Quick verification script."""
import sys
sys.path.insert(0, '.')

# Test all imports
from src.config import BODY_25_KEYPOINTS, SKELETON_CONNECTIONS
from src.data.loader import load_openpose_json, load_csv_sequence, normalize_keypoints
from src.features.kinematic import extract_kinematic_features
from src.features.statistical import extract_all_features
from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer, LSTMClassifier, TransformerClassifier
from src.models.training import run_full_comparison, save_results
from src.visualization.plots import plot_pose_skeleton, create_interactive_skeleton_html
from app.utils import load_all_models, get_ensemble_prediction, format_prediction_result
print('All imports successful!')

# Verify models were saved
import os
models_dir = 'models'
files_to_check = ['rf_model.pkl', 'svm_model.pkl', 'lstm_model.pth', 'transformer_model.pth', 'comparison_results.json']
for f in files_to_check:
    path = os.path.join(models_dir, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  {f}: {status} ({size:,} bytes)")

# Test loading models
models = load_all_models(models_dir)
print(f"\nLoaded {len(models)} models: {list(models.keys())}")

# Test inference on synthetic data
import numpy as np
rng = np.random.RandomState(42)
keypoints = rng.rand(30, 25, 2).astype(np.float32)

kinematic_feats, kinematic_names = extract_kinematic_features(keypoints)
stat_feats, stat_names = extract_all_features(keypoints)
all_features = np.concatenate([kinematic_feats, stat_feats])
dl_sequence = keypoints.reshape(30, -1)

if len(models) > 0:
    predictions = get_ensemble_prediction(models, all_features, dl_sequence)
    formatted = format_prediction_result(predictions)
    print(f"\nEnsemble ASD Probability: {formatted['ensemble_probability']:.1%}")
    print(f"Risk Level: {formatted['risk_level']}")
    for model_name, model_data in formatted['model_predictions'].items():
        print(f"  {model_name}: {model_data['probability']:.1%} ({model_data['risk_level']})")

print("\nVerification complete!")

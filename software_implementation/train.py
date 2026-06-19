"""
Training script for the autism screening pipeline.

Usage:
    python train.py --data_dir /path/to/mmasd --epochs 30 --cv_folds 5
    python train.py --synthetic --n_samples 100  # For testing without real data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_dataset_from_csv,
    normalize_keypoints,
    pad_or_truncate_sequence,
)
from src.features.kinematic import extract_kinematic_features
from src.features.statistical import extract_all_features
from src.models.training import run_full_comparison, save_results


def generate_synthetic_data(n_samples: int = 100, seq_length: int = 50,
                            n_joints: int = 25, seed: int = 42) -> tuple:
    """
    Generate synthetic keypoint data for testing the pipeline.

    Creates two classes with subtly different movement patterns.

    Args:
        n_samples: Number of synthetic subjects
        seq_length: Number of frames per sequence
        n_joints: Number of joints (25 for BODY_25)
        seed: Random seed

    Returns:
        X_features: Feature matrix for ML models
        y: Labels (0 or 1)
        X_sequences: Sequences for DL models
        feature_names: List of feature names
    """
    rng = np.random.RandomState(seed)

    all_features = []
    all_sequences = []
    all_labels = []
    all_feature_names = None

    for i in range(n_samples):
        label = i % 2  # Balanced classes

        # Generate synthetic keypoint sequence
        # Class 0 (TD): Smooth, coordinated movements
        # Class 1 (ASD): More erratic, asymmetric movements
        base_freq = rng.uniform(0.5, 2.0)
        amplitude = rng.uniform(0.1, 0.3)

        if label == 0:
            # Smooth sinusoidal movements
            t = np.linspace(0, 2 * np.pi, seq_length)
            keypoints = np.zeros((seq_length, n_joints, 2), dtype=np.float32)
            for j in range(n_joints):
                phase = rng.uniform(0, 2 * np.pi)
                keypoints[:, j, 0] = 0.5 + amplitude * np.sin(base_freq * t + phase)
                keypoints[:, j, 1] = 0.5 + amplitude * np.cos(base_freq * t + phase * 0.7)
        else:
            # Erratic movements with higher frequency components
            t = np.linspace(0, 2 * np.pi, seq_length)
            keypoints = np.zeros((seq_length, n_joints, 2), dtype=np.float32)
            for j in range(n_joints):
                phase = rng.uniform(0, 2 * np.pi)
                # Add noise and higher frequency components
                keypoints[:, j, 0] = (0.5 + amplitude * np.sin(base_freq * t + phase)
                                     + 0.1 * np.sin(3 * base_freq * t + phase))
                keypoints[:, j, 1] = (0.5 + amplitude * np.cos(base_freq * t + phase * 0.7)
                                     + 0.1 * np.cos(2.5 * base_freq * t + phase))
                # Add more noise for ASD class
                keypoints[:, j, 0] += rng.normal(0, 0.02, seq_length)
                keypoints[:, j, 1] += rng.normal(0, 0.02, seq_length)

        # Extract features
        kinematic_feats, kinematic_names = extract_kinematic_features(keypoints)
        stat_feats, stat_names = extract_all_features(keypoints)

        all_feats = np.concatenate([kinematic_feats, stat_feats])
        if all_feature_names is None:
            all_feature_names = kinematic_names + stat_names

        all_features.append(all_feats)
        all_sequences.append(keypoints.reshape(seq_length, -1))  # (frames, joints*coords)
        all_labels.append(label)

    X_features = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    X_sequences = all_sequences

    return X_features, y, X_sequences, all_feature_names


def train_with_real_data(data_dir: str, models_dir: str,
                         cv_folds: int = 5, dl_epochs: int = 30,
                         target_length: int = 50) -> None:
    """
    Train models using real MMASD dataset.

    Args:
        data_dir: Path to MMASD data directory
        models_dir: Path to save trained models
        cv_folds: Number of cross-validation folds
        dl_epochs: Training epochs for DL models
        target_length: Fixed sequence length for DL models
    """
    print(f"Loading dataset from {data_dir}...")

    subjects, labels = load_dataset_from_csv(data_dir)

    print(f"Loaded {len(subjects)} subjects")
    print(f"Class distribution: TD={np.sum(labels == 0)}, ASD={np.sum(labels == 1)}")

    # Extract features for ML models
    all_features = []
    all_sequences = []
    feature_names = None

    for subject in subjects:
        keypoints = subject['keypoints']

        # For ML: extract flat features from full sequence
        coords_2d = keypoints[:, :, :2]  # Use x, y only
        kinematic_feats, kinematic_names = extract_kinematic_features(coords_2d)
        stat_feats, stat_names = extract_all_features(coords_2d)

        all_feats = np.concatenate([kinematic_feats, stat_feats])
        if feature_names is None:
            feature_names = kinematic_names + stat_names

        all_features.append(all_feats)

        # For DL: pad/truncate to fixed length
        seq = keypoints.reshape(keypoints.shape[0], -1)  # (frames, joints*coords)
        seq_padded = pad_or_truncate_sequence(seq.reshape(-1, keypoints.shape[1], keypoints.shape[2]), target_length)
        seq_flat = seq_padded.reshape(target_length, -1)
        all_sequences.append(seq_flat)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

    # Run full comparison
    results = run_full_comparison(
        X, y, all_sequences, feature_names,
        cv_folds=cv_folds, dl_epochs=dl_epochs
    )

    # Save results
    os.makedirs(models_dir, exist_ok=True)
    save_results(results, models_dir)


def main():
    parser = argparse.ArgumentParser(description="Train autism screening models")
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to MMASD dataset directory')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs for DL models')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of synthetic samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    if args.synthetic or args.data_dir is None:
        print("Using synthetic data for testing...")
        X, y, X_sequences, feature_names = generate_synthetic_data(
            n_samples=args.n_samples, seed=args.seed
        )

        print(f"Synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: TD={np.sum(y == 0)}, ASD={np.sum(y == 1)}")

        results = run_full_comparison(
            X, y, X_sequences, feature_names,
            cv_folds=min(args.cv_folds, min(np.sum(y == 0), np.sum(y == 1))),
            dl_epochs=args.epochs,
        )

        os.makedirs(args.models_dir, exist_ok=True)
        save_results(results, args.models_dir)
    else:
        train_with_real_data(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            cv_folds=args.cv_folds,
            dl_epochs=args.epochs,
        )


if __name__ == '__main__':
    main()

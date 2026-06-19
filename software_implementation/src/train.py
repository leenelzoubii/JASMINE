#!/usr/bin/env python3
"""
Train all models (RF, SVM, LSTM, Transformer) and produce comparison.
Optimized training loop with hyperparameter tuning, early stopping,
gradient clipping, LR warmup, and weighted ensemble.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import (
    load_dataset_from_csv,
    normalize_keypoints,
    pad_or_truncate_sequence,
)
from src.features.kinematic import extract_kinematic_features
from src.features.statistical import extract_all_features
from src.models.training import run_full_comparison, save_results


def train_with_data(data_dir: str, output_dir: str = 'results',
                    cv_folds: int = 5, dl_epochs: int = 100,
                    enable_feature_selection: bool = True,
                    target_length: int = 50) -> None:
    print(f"Loading dataset from {data_dir}...")
    subjects, labels = load_dataset_from_csv(data_dir)

    print(f"Loaded {len(subjects)} subjects")
    print(f"Class distribution: TD={sum(1 for l in labels if l == 0)}, ASD={sum(1 for l in labels if l == 1)}")

    all_features = []
    all_sequences = []
    feature_names = None

    for subject in subjects:
        keypoints = subject['keypoints']
        coords_2d = keypoints[:, :, :2]
        kinematic_feats, kinematic_names = extract_kinematic_features(coords_2d)
        stat_feats, stat_names = extract_all_features(coords_2d)
        all_feats = np.concatenate([kinematic_feats, stat_feats])
        if feature_names is None:
            feature_names = kinematic_names + stat_names
        all_features.append(all_feats)
        seq = keypoints.reshape(keypoints.shape[0], -1)
        seq_padded = pad_or_truncate_sequence(seq.reshape(-1, keypoints.shape[1], keypoints.shape[2]), target_length)
        seq_flat = seq_padded.reshape(target_length, -1)
        all_sequences.append(seq_flat)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

    cv_results = run_full_comparison(
        X, y, all_sequences, feature_names,
        cv_folds=cv_folds,
        dl_epochs=dl_epochs,
        enable_feature_selection=enable_feature_selection,
    )

    save_results(cv_results, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Train autism screening models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs for deep learning models')
    parser.add_argument('--folds', type=int, default=5,
                        help='Cross-validation folds')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection for ML models')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--data-dir', type=str, default='data/csv',
                        help='Data directory containing CSV files')
    args = parser.parse_args()

    print("=" * 60)
    print("AUTISM SCREENING - OPTIMIZED MODEL TRAINING")
    print("=" * 60)

    train_with_data(
        data_dir=args.data_dir,
        output_dir=args.output,
        cv_folds=args.folds,
        dl_epochs=args.epochs,
        enable_feature_selection=not args.no_feature_selection,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

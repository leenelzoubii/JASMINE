"""Tests for feature extraction modules."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.kinematic import (
    compute_angle,
    compute_joint_angles,
    compute_joint_velocities,
    compute_inter_joint_distances,
    extract_kinematic_features,
)
from src.features.statistical import (
    extract_keypoint_stats,
    extract_temporal_features,
    extract_frequency_features,
    extract_all_features,
)


def _create_test_sequence(num_frames: int = 50, num_joints: int = 25) -> np.ndarray:
    """Create a test keypoint sequence."""
    rng = np.random.RandomState(42)
    keypoints = np.zeros((num_frames, num_joints, 2), dtype=np.float32)

    # Create some structured movement
    t = np.linspace(0, 2 * np.pi, num_frames)
    for j in range(num_joints):
        keypoints[:, j, 0] = 0.5 + 0.2 * np.sin(t + j * 0.1)
        keypoints[:, j, 1] = 0.5 + 0.15 * np.cos(t + j * 0.15)

    return keypoints


class TestKinematicFeatures:
    """Tests for kinematic feature extraction."""

    def test_compute_angle(self):
        """Test angle computation between three points."""
        # Right angle: (0,0) -> (1,0) -> (1,1)
        p1 = np.array([[0.0, 0.0]])
        p2 = np.array([[1.0, 0.0]])
        p3 = np.array([[1.0, 1.0]])

        angle = compute_angle(p1, p2, p3)
        assert np.isclose(angle[0], np.pi / 2, atol=1e-6)

    def test_compute_joint_angles(self):
        """Test joint angle computation for a sequence."""
        keypoints = _create_test_sequence()
        angles = compute_joint_angles(keypoints)

        assert angles.shape[0] == keypoints.shape[0]  # frames
        assert angles.shape[1] == 10  # number of triplets
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi)

    def test_compute_joint_velocities(self):
        """Test joint velocity computation."""
        keypoints = _create_test_sequence()
        velocities = compute_joint_velocities(keypoints, fps=30)

        assert velocities.shape[0] == keypoints.shape[0] - 1  # one less frame
        assert velocities.shape[1] == keypoints.shape[1]  # all joints
        assert np.all(velocities >= 0)

    def test_compute_inter_joint_distances(self):
        """Test inter-joint distance computation."""
        keypoints = _create_test_sequence()
        distances = compute_inter_joint_distances(keypoints)

        assert distances.shape[0] == keypoints.shape[0]
        assert distances.shape[1] == 8  # number of distance pairs
        assert np.all(distances >= 0)

    def test_extract_kinematic_features(self):
        """Test full kinematic feature extraction."""
        keypoints = _create_test_sequence()
        features, names = extract_kinematic_features(keypoints)

        assert len(features) > 0
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestStatisticalFeatures:
    """Tests for statistical feature extraction."""

    def test_extract_keypoint_stats(self):
        """Test keypoint statistical feature extraction."""
        keypoints = _create_test_sequence()
        stats, names = extract_keypoint_stats(keypoints)

        assert len(stats) > 0
        assert len(stats) == len(names)
        assert not np.any(np.isnan(stats))

    def test_extract_temporal_features(self):
        """Test temporal feature extraction."""
        keypoints = _create_test_sequence()
        features, names = extract_temporal_features(keypoints)

        assert len(features) > 0
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))

    def test_extract_frequency_features(self):
        """Test frequency domain feature extraction."""
        keypoints = _create_test_sequence()
        features, names = extract_frequency_features(keypoints)

        assert len(features) > 0
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))

    def test_extract_all_features(self):
        """Test combined feature extraction."""
        keypoints = _create_test_sequence()
        features, names = extract_all_features(keypoints)

        assert len(features) > 0
        assert len(features) == len(names)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestFeatureConsistency:
    """Test consistency between different feature extraction methods."""

    def test_feature_shapes_consistent(self):
        """Test that feature extraction produces consistent shapes."""
        keypoints = _create_test_sequence(num_frames=50, num_joints=25)

        kinematic_feats, kinematic_names = extract_kinematic_features(keypoints)
        stat_feats, stat_names = extract_all_features(keypoints)

        # Both should produce non-empty feature vectors
        assert len(kinematic_feats) > 0
        assert len(stat_feats) > 0

        # Names should match feature counts
        assert len(kinematic_feats) == len(kinematic_names)
        assert len(stat_feats) == len(stat_names)

    def test_deterministic_output(self):
        """Test that feature extraction is deterministic."""
        keypoints = _create_test_sequence()

        features1, _ = extract_all_features(keypoints)
        features2, _ = extract_all_features(keypoints)

        assert np.allclose(features1, features2)

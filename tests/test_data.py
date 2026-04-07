"""Tests for data loading module."""

import os
import sys
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.loader import (
    load_openpose_json,
    load_openpose_sequence,
    load_csv_sequence,
    normalize_keypoints,
    pad_or_truncate_sequence,
)


def _create_openpose_json(path: str, num_joints: int = 25):
    """Create a test OpenPose JSON file."""
    keypoints = []
    for j in range(num_joints):
        keypoints.extend([float(j * 10), float(j * 5 + 100), 0.9])

    data = {
        "version": 1.1,
        "people": [{
            "pose_keypoints_2d": keypoints,
        }]
    }
    with open(path, 'w') as f:
        json.dump(data, f)


def _create_csv_file(path: str, num_frames: int = 30, num_joints: int = 25):
    """Create a test MMASD-style CSV file."""
    import pandas as pd

    columns = []
    joint_names = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                   "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                   "left_wrist", "right_wrist", "left_hip", "right_hip",
                   "left_knee", "right_knee", "left_ankle", "right_ankle"]

    # Use 17 joints (MediaPipe style) for CSV test
    for j in range(min(num_joints, 17)):
        name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    # Add label columns
    columns.extend(["Action_Label", "ASD_Label"])

    # Generate random data
    data = np.random.rand(num_frames, len(columns)).astype(np.float32)
    data[:, -2] = 0  # Action_Label
    data[:, -1] = np.random.choice([0, 1], num_frames).astype(np.float32)  # ASD_Label

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)


class TestOpenPoseLoader:
    """Tests for OpenPose JSON loading."""

    def test_load_single_json(self):
        """Test loading a single OpenPose JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name

        try:
            _create_openpose_json(tmp_path)
            keypoints = load_openpose_json(tmp_path)

            assert keypoints.shape == (25, 3)
            assert keypoints.dtype == np.float32
            assert keypoints[0, 0] == 0.0   # Nose x
            assert keypoints[0, 1] == 100.0  # Nose y
            assert keypoints[0, 2] == 0.9    # Nose confidence
        finally:
            os.unlink(tmp_path)

    def test_load_empty_json(self):
        """Test loading JSON with no people detected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"version": 1.1, "people": []}, f)
            tmp_path = f.name

        try:
            keypoints = load_openpose_json(tmp_path)
            assert keypoints.shape == (25, 3)
            assert np.all(keypoints == 0)
        finally:
            os.unlink(tmp_path)

    def test_load_sequence(self):
        """Test loading a sequence of OpenPose JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 5 JSON files
            for i in range(5):
                _create_openpose_json(os.path.join(tmpdir, f'frame_{i:04d}.json'))

            sequence = load_openpose_sequence(tmpdir)
            assert sequence.shape == (5, 25, 3)
            assert sequence.dtype == np.float32


class TestCSVLoader:
    """Tests for MMASD CSV loading."""

    def test_load_csv_sequence(self):
        """Test loading a MMASD-style CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            tmp_path = f.name

        try:
            _create_csv_file(tmp_path, num_frames=30, num_joints=17)
            keypoints, action_label, asd_label = load_csv_sequence(tmp_path)

            assert keypoints.shape[0] == 30  # frames
            assert keypoints.shape[1] == 17  # joints
            assert keypoints.shape[2] == 3   # x, y, z
            assert action_label is not None
            assert asd_label is not None
        finally:
            os.unlink(tmp_path)


class TestNormalization:
    """Tests for keypoint normalization."""

    def test_normalize_keypoints(self):
        """Test min-max normalization of keypoints."""
        keypoints = np.array([[[0, 0, 0.9], [100, 100, 0.8]],
                              [[50, 50, 0.7], [150, 150, 0.6]]], dtype=np.float32)

        normalized = normalize_keypoints(keypoints)

        # Check range
        assert normalized[..., 0].min() >= 0
        assert normalized[..., 0].max() <= 1
        assert normalized[..., 1].min() >= 0
        assert normalized[..., 1].max() <= 1

        # Confidence should be preserved
        assert np.allclose(normalized[..., 2], keypoints[..., 2])

    def test_normalize_constant_keypoints(self):
        """Test normalization when all keypoints are the same."""
        keypoints = np.ones((10, 25, 3), dtype=np.float32)
        normalized = normalize_keypoints(keypoints)

        # Should not raise or produce NaN
        assert not np.any(np.isnan(normalized))


class TestPadding:
    """Tests for sequence padding/truncation."""

    def test_pad_sequence(self):
        """Test padding a short sequence."""
        keypoints = np.random.rand(10, 25, 3).astype(np.float32)
        padded = pad_or_truncate_sequence(keypoints, target_length=30)

        assert padded.shape == (30, 25, 3)
        # Original data preserved
        assert np.allclose(padded[:10], keypoints)
        # Padding is zeros
        assert np.allclose(padded[10:], 0)

    def test_truncate_sequence(self):
        """Test truncating a long sequence."""
        keypoints = np.random.rand(100, 25, 3).astype(np.float32)
        truncated = pad_or_truncate_sequence(keypoints, target_length=30)

        assert truncated.shape == (30, 25, 3)

    def test_exact_length(self):
        """Test sequence that matches target length."""
        keypoints = np.random.rand(30, 25, 3).astype(np.float32)
        result = pad_or_truncate_sequence(keypoints, target_length=30)

        assert result.shape == (30, 25, 3)
        assert np.allclose(result, keypoints)

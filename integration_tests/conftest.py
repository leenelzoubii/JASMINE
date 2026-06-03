"""
Pytest configuration for integration tests.
Sets up paths and shared fixtures for end-to-end testing.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def synthetic_keypoints():
    """Generate synthetic keypoint sequence (realistic movement)."""
    num_frames = 100
    num_joints = 25
    
    # Create structured movement pattern (simulating arm swinging)
    t = np.linspace(0, 4 * np.pi, num_frames)
    keypoints = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    
    # Simulate body movement
    for j in range(num_joints):
        # X: horizontal movement
        keypoints[:, j, 0] = 0.5 + 0.15 * np.sin(t + j * 0.2)
        # Y: vertical movement
        keypoints[:, j, 1] = 0.5 + 0.1 * np.cos(t + j * 0.15)
        # Z: confidence/visibility
        keypoints[:, j, 2] = 0.8 + 0.2 * np.random.rand()
    
    return keypoints


@pytest.fixture
def synthetic_video_path(tmp_path):
    """Generate a synthetic MP4 video file."""
    try:
        import cv2
    except ImportError:
        pytest.skip("OpenCV not available for video generation")
    
    video_path = tmp_path / "test_video.mp4"
    
    # Video parameters
    fps = 30
    width, height = 640, 480
    num_frames = 90
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Generate frames with moving circle (simulating person moving)
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw moving circle
        center_x = int(width / 2 + 100 * np.sin(2 * np.pi * i / num_frames))
        center_y = int(height / 2 + 50 * np.cos(2 * np.pi * i / num_frames))
        cv2.circle(frame, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # Write frame
        out.write(frame)
    
    out.release()
    return str(video_path)


@pytest.fixture
def test_features():
    """Generate realistic feature vector."""
    # Typical feature count: ~175 kinematic + ~750 statistical = ~925
    num_features = 150  # Simplified for testing
    features = np.random.randn(num_features).astype(np.float32)
    return features


@pytest.fixture
def sample_csv_path(tmp_path):
    """Generate a sample MMASD-style CSV file."""
    try:
        import pandas as pd
    except ImportError:
        pytest.skip("Pandas not available")
    
    csv_path = tmp_path / "sample_data.csv"
    
    # Create realistic CSV structure
    num_frames = 50
    num_joints = 17
    
    columns = []
    for j in range(num_joints):
        columns.extend([f"joint_{j}_x", f"joint_{j}_y", f"joint_{j}_z"])
    
    columns.extend(["Action_Label", "ASD_Label"])
    
    # Generate data with realistic ranges
    data = np.random.rand(num_frames, len(columns)).astype(np.float32)
    data[:, :-2] = data[:, :-2] * 0.8 + 0.1  # Normalize to ~[0.1, 0.9]
    data[:, -2] = 0  # Action label
    data[:, -1] = np.random.choice([0, 1], num_frames)  # ASD label
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


@pytest.fixture
def cleanup_temp_files():
    """Cleanup temporary test files after test."""
    yield
    # Cleanup happens automatically with tmp_path, but can add custom cleanup here

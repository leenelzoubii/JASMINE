"""
MMASD dataset loader.

Supports both the Enhanced-MMASD CSV format (MediaPipe 3D skeleton)
and the original OpenPose JSON format (BODY_25 2D keypoints).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# MMASD+ CSV column mapping (26 joints × 3 coords = 78 columns)
# Joint names from the Enhanced-MMASD format
MMASD_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot", "right_foot",
]

NUM_MMASD_JOINTS = 25  # 26 joints but we use 25 (matching BODY_25)

# Action label mapping from MMASD+ dataset
ACTION_TO_LABEL = {
    'Arm': 0, 'bs': 1, 'ce': 2, 'dr': 3, 'fg': 4,
    'mfs': 5, 'ms': 6, 'sq': 7, 'tw': 8, 'sac': 9, 'tr': 10
}


def load_csv_sequence(csv_path: str) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    df = pd.read_csv(csv_path)

    # Extract labels if present
    action_label = None
    asd_label = None
    if 'Action_Label' in df.columns:
        action_label = int(df['Action_Label'].iloc[0])
        df = df.drop(columns=['Action_Label'])
    if 'ASD_Label' in df.columns:
        asd_label = int(df['ASD_Label'].iloc[0])
        df = df.drop(columns=['ASD_Label'])

    # 🧼 نظف البيانات
    df = df.fillna(0)

    # Extract coordinate columns
    coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y') or c.endswith('_z')]

    try:
        # 🟢 الحالة 1: MMASD format
        if len(coord_cols) > 0:
            coords = df[coord_cols].values.astype(np.float32)
            num_frames = coords.shape[0]
            num_joints = len(coord_cols) // 3

            keypoints = coords.reshape(num_frames, num_joints, 3)

        # 🟡 الحالة 2: CSV عادي
        else:
            coords = df.values.astype(np.float32)
            num_frames = coords.shape[0]
            num_joints = coords.shape[1] // 3

            keypoints = coords.reshape(num_frames, num_joints, 3)

        # 🔥 أهم خطوة: خذ فقط x,y
        keypoints = keypoints[:, :, :2]

        return keypoints, action_label if action_label is not None else "unknown", \
               asd_label if asd_label is not None else -1

    except Exception as e:
        print(f"Error inside loader: {e}")
        return None, None, None  


def load_openpose_json(json_path: str) -> np.ndarray:
    """
    Load a single OpenPose JSON file (one frame).

    Args:
        json_path: Path to the OpenPose JSON file

    Returns:
        keypoints: numpy array of shape (25, 3) for BODY_25
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data.get('people'):
        return np.zeros((25, 3), dtype=np.float32)

    person = data['people'][0]
    keypoints_2d = person.get('pose_keypoints_2d', [])

    if len(keypoints_2d) == 0:
        return np.zeros((25, 3), dtype=np.float32)

    keypoints = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)
    return keypoints


def load_openpose_sequence(directory: str) -> np.ndarray:
    """
    Load all OpenPose JSON files from a directory as a sequence.

    Args:
        directory: Path to directory containing OpenPose JSON files

    Returns:
        keypoints: numpy array of shape (frames, 25, 3)
    """
    json_files = sorted(Path(directory).glob('*.json'))

    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")

    sequences = []
    for jf in json_files:
        kp = load_openpose_json(str(jf))
        sequences.append(kp)

    return np.array(sequences, dtype=np.float32)


def parse_subject_id_from_csv(filename: str) -> str:
    """Extract subject ID from MMASD+ CSV filename."""
    # Format: processed_{action}_{subject_id}_{session}_{recording}_{gender}_{age}_{asd}.csv
    parts = Path(filename).stem.split('_')
    if len(parts) >= 3:
        return parts[2]
    return Path(filename).stem


def load_subject_from_csv(csv_path: str) -> Dict:
    """
    Load a single subject from a CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        dict with keys: keypoints, action_label, asd_label, subject_id, num_frames
    """
    keypoints, action_label, asd_label = load_csv_sequence(csv_path)

    return {
        'keypoints': keypoints,
        'action_label': action_label,
        'asd_label': asd_label,
        'subject_id': parse_subject_id_from_csv(csv_path),
        'num_frames': keypoints.shape[0],
    }


def load_dataset_from_csv(data_dir: str, asd_label_only: bool = True) -> Tuple[List[Dict], np.ndarray]:
    """
    Load the entire MMASD+ dataset from CSV files.

    Args:
        data_dir: Root directory containing action subdirectories
        asd_label_only: If True, return only ASD labels; if False, return action labels

    Returns:
        subjects: List of subject dicts
        labels: numpy array of labels
    """
    data_path = Path(data_dir)
    subjects = []
    labels = []

    # Search recursively for CSV files
    csv_files = list(data_path.rglob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    for csv_file in csv_files:
        try:
            subject = load_subject_from_csv(str(csv_file))
            if asd_label_only:
                if subject['asd_label'] is not None:
                    subjects.append(subject)
                    labels.append(subject['asd_label'])
            else:
                if subject['action_label'] is not None:
                    subjects.append(subject)
                    labels.append(subject['action_label'])
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue

    if not subjects:
        raise ValueError(f"No valid subjects found in {data_dir}")

    return subjects, np.array(labels, dtype=np.int64)


def load_dataset_from_openpose(data_dir: str, metadata_path: str) -> Tuple[List[Dict], np.ndarray]:
    """
    Load dataset from OpenPose JSON directories with Excel metadata.

    Args:
        data_dir: Root directory containing subject subdirectories with JSON files
        metadata_path: Path to Excel file with subject metadata

    Returns:
        subjects: List of subject dicts
        labels: numpy array of ASD labels
    """
    metadata = pd.read_excel(metadata_path)
    subjects = []
    labels = []

    data_path = Path(data_dir)

    for _, row in metadata.iterrows():
        subject_id = str(row.get('subject_id', row.iloc[0]))
        subject_dir = data_path / subject_id

        if not subject_dir.exists():
            print(f"Warning: Subject directory not found: {subject_dir}")
            continue

        try:
            keypoints = load_openpose_sequence(str(subject_dir))
            label = int(row.get('ASD_Label', row.get('label', 0)))

            subjects.append({
                'keypoints': keypoints,
                'asd_label': label,
                'subject_id': subject_id,
                'num_frames': keypoints.shape[0],
                'metadata': row.to_dict(),
            })
            labels.append(label)
        except Exception as e:
            print(f"Warning: Failed to load subject {subject_id}: {e}")
            continue

    return subjects, np.array(labels, dtype=np.int64)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints to [0, 1] range using min-max scaling.

    Args:
        keypoints: numpy array of shape (frames, joints, coords)

    Returns:
        Normalized keypoints
    """
    coords = keypoints[..., :2]  # Only x, y for 2D
    mins = coords.min(axis=(0, 1), keepdims=True)
    maxs = coords.max(axis=(0, 1), keepdims=True)

    # Avoid division by zero
    range_vals = maxs - mins
    range_vals[range_vals == 0] = 1.0

    normalized = (coords - mins) / range_vals

    # Keep confidence as-is
    if keypoints.shape[-1] == 3:
        normalized = np.concatenate([normalized, keypoints[..., 2:3]], axis=-1)

    return normalized


def pad_or_truncate_sequence(keypoints: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate a keypoint sequence to a fixed length.

    Args:
        keypoints: numpy array of shape (frames, joints, coords)
        target_length: Desired number of frames

    Returns:
        numpy array of shape (target_length, joints, coords)
    """
    current_length = keypoints.shape[0]

    if current_length >= target_length:
        # Truncate: take evenly spaced frames
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return keypoints[indices]
    else:
        # Pad with zeros
        pad_width = ((0, target_length - current_length), (0, 0), (0, 0))
        return np.pad(keypoints, pad_width, mode='constant', constant_values=0)

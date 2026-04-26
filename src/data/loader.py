"""
MMASD dataset loader.

Supports both the Enhanced-MMASD CSV format (MediaPipe 3D skeleton)
and the original OpenPose JSON format (BODY_25 2D keypoints).

Enhanced with multi-person tracking support:
- load_all_people_from_openpose_json: Loads multiple people from a single frame
- classify_person_by_size: Classifies person as adult/child based on skeletal size
- extract_child_keypoints: Extracts child skeleton for prediction
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# MULTI-PERSON TRACKING CONFIGURATION
# ============================================================

# Height threshold (normalized coordinates) - person below this is likely a child
# Default: 0.35 works well for normalized [0,1] coordinates
# Adjust based on your specific video setup!
CHILD_HEIGHT_THRESHOLD = 0.35

# Bounding box padding (as fraction of body size)
BB_PADDING = 0.15

# Person classification colors (for visualization)
PERSON_COLORS = {
    'instructor': '#2ca02c',  # Green
    'child': '#d62728',        # Red
    'unknown': '#ff7f0e'      # Orange
}

PERSON_LABELS = {
    'instructor': 'Instructor',
    'child': 'Child/Subject',
    'unknown': 'Person'
}

# Size-based heuristics for person classification
SIZE_FEATURES = {
    'height': 0,      # nose to midhip distance
    'arm_span': 1,      # wrist to wrist distance  
    'shoulder_width': 2, # shoulder width
    'hip_width': 3       # hip width
}


# ============================================================
# MULTI-PERSON TRACKING FUNCTIONS
# ============================================================

def calculate_person_size(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Calculate size metrics for a person from keypoints.
    
    Args:
        keypoints: numpy array of shape (joints, 3) or (frames, joints, 3)
        
    Returns:
        dict with size metrics: height, arm_span, shoulder_width, hip_width
    """
    # Use first frame if multiple frames
    if keypoints.ndim == 3:
        kp = keypoints[0]
    else:
        kp = keypoints
    
    coords = kp[:, :2]  # x, y only
    
    metrics = {}
    
    # Height: nose (0) to midhip (8)
    if coords[0, 1] > 0 and coords[8, 1] > 0:
        metrics['height'] = abs(coords[0, 1] - coords[8, 1])
    else:
        metrics['height'] = 0
    
    # Arm span: left wrist (7) to right wrist (4)
    if coords[7, 0] > 0 and coords[4, 0] > 0:
        metrics['arm_span'] = abs(coords[7, 0] - coords[4, 0])
    else:
        metrics['arm_span'] = 0
    
    # Shoulder width: left shoulder (5) to right shoulder (2)
    if coords[5, 0] > 0 and coords[2, 0] > 0:
        metrics['shoulder_width'] = abs(coords[5, 0] - coords[2, 0])
    else:
        metrics['shoulder_width'] = 0
    
    # Hip width: left hip (12) to right hip (9)
    if coords[12, 0] > 0 and coords[9, 0] > 0:
        metrics['hip_width'] = abs(coords[12, 0] - coords[9, 0])
    else:
        metrics['hip_width'] = 0
    
    return metrics


def classify_person_by_size(keypoints: np.ndarray, 
                          height_threshold: float = CHILD_HEIGHT_THRESHOLD) -> str:
    """
    Classify a person as 'instructor' or 'child' based on skeletal size.
    
    Args:
        keypoints: numpy array of shape (joints, 3) or (frames, joints, 3)
        height_threshold: Height threshold for classification
        
    Returns:
        'instructor' if adult (taller), 'child' if shorter
    """
    size_metrics = calculate_person_size(keypoints)
    height = size_metrics['height']
    
    if height > height_threshold:
        return 'instructor'
    else:
        return 'child'


def calculate_bounding_box(keypoints: np.ndarray, 
                        padding: float = BB_PADDING) -> Tuple[float, float, float, float]:
    """
    Calculate bounding box for a person's keypoints.
    
    Args:
        keypoints: numpy array of shape (joints, 3) or (frames, joints, 3)
        padding: Padding fraction around the bounding box
        
    Returns:
        (x_min, y_min, x_max, y_max) tuple
    """
    # Use first frame if multiple frames
    if keypoints.ndim == 3:
        kp = keypoints[0]
    else:
        kp = keypoints
    
    coords = kp[:, :2]
    
    # Find valid (non-zero) points
    valid_mask = (coords[:, 0] > 0) | (coords[:, 1] > 0)
    
    if not np.any(valid_mask):
        return 0.0, 0.0, 1.0, 1.0
    
    valid_coords = coords[valid_mask]
    
    x_min = valid_coords[:, 0].min()
    x_max = valid_coords[:, 0].max()
    y_min = valid_coords[:, 1].min()
    y_max = valid_coords[:, 1].max()
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding
    
    return x_min, y_min, x_max, y_max


def load_all_people_from_openpose_json(json_path: str) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Load ALL people detected in a single OpenPose JSON frame.
    
    Args:
        json_path: Path to the OpenPose JSON file
        
    Returns:
        Tuple of:
        - keypoints_list: List of numpy arrays, each (25, 3) for BODY_25
        - person_info_list: List of dicts with person metadata
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    people = data.get('people', [])
    
    if not people:
        # Return empty single person structure for backward compatibility
        return [np.zeros((25, 3), dtype=np.float32)], [{
            'person_id': 0,
            'classification': 'unknown',
            'size_metrics': {},
            'bounding_box': (0, 0, 1, 1)
        }]
    
    keypoints_list = []
    person_info_list = []
    
    for idx, person in enumerate(people):
        keypoints_2d = person.get('pose_keypoints_2d', [])
        
        if len(keypoints_2d) == 0:
            continue
        
        keypoints = np.array(keypoints_2d, dtype=np.float32).reshape(-1, 3)
        
        # Skip if not BODY_25 (at least 25 joints)
        if keypoints.shape[0] < 25:
            continue
        
        # Trim to 25 joints
        keypoints = keypoints[:25]
        
        # Calculate person info
        size_metrics = calculate_person_size(keypoints)
        classification = classify_person_by_size(keypoints)
        bounding_box = calculate_bounding_box(keypoints)
        
        person_info = {
            'person_id': idx,
            'classification': classification,
            'size_metrics': size_metrics,
            'bounding_box': bounding_box
        }
        
        keypoints_list.append(keypoints)
        person_info_list.append(person_info)
    
    # If no valid people found, return empty structure
    if not keypoints_list:
        return [np.zeros((25, 3), dtype=np.float32)], [{
            'person_id': 0,
            'classification': 'unknown',
            'size_metrics': {},
            'bounding_box': (0, 0, 1, 1)
        }]
    
    return keypoints_list, person_info_list


def extract_child_keypoints(keypoints_list: List[np.ndarray], 
                       person_info_list: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Extract the child/subject keypoints from a list of detected people.
    
    If no person is classified as 'child', returns the smallest person.
    
    Args:
        keypoints_list: List of keypoint arrays
        person_info_list: List of person info dicts
        
    Returns:
        Tuple of (child_keypoints, child_info) or (None, None) if not found
    """
    # First, try to find a person classified as 'child'
    for kp, info in zip(keypoints_list, person_info_list):
        if info['classification'] == 'child':
            return kp, info
    
    # Fallback: return the smallest person (likely the child in the scene)
    smallest_height = float('inf')
    child_idx = -1
    
    for idx, info in enumerate(person_info_list):
        height = info['size_metrics'].get('height', 0)
        if 0 < height < smallest_height:
            smallest_height = height
            child_idx = idx
    
    if child_idx >= 0:
        return keypoints_list[child_idx], person_info_list[child_idx]
    
    return None, None


def load_openpose_sequence_with_multi_person(directory: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load OpenPose JSON sequence with MULTIPLE people tracked per frame.
    
    Args:
        directory: Path to directory containing OpenPose JSON files
        
    Returns:
        Tuple of:
        - keypoints_sequence: numpy array of shape (frames, num_people, 25, 3)
        - all_person_info: List of person info per frame
    """
    json_files = sorted(Path(directory).glob('*.json'))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")
    
    all_frames_people = []
    all_person_info = []
    
    for jf in json_files:
        try:
            kp_list, info_list = load_all_people_from_openpose_json(str(jf))
            all_frames_people.append(kp_list)
            all_person_info.append(info_list)
        except Exception as e:
            print(f"Warning: Failed to load {jf}: {e}")
            continue
    
    # Pad to same number of people per frame
    max_people = max(len(frame_people) for frame_people in all_frames_people) if all_frames_people else 1
    
    # Create sequence array: (frames, people, joints, coords)
    num_frames = len(all_frames_people)
    
    # Use first person only for backward compatibility
    sequence = np.zeros((num_frames, 25, 3), dtype=np.float32)
    for frame_idx, frame_people in enumerate(all_frames_people):
        if frame_people:
            sequence[frame_idx] = frame_people[0]
    
    return sequence, all_person_info


# ============================================================
# LEGACY FUNCTIONS (Backward Compatibility)
# ============================================================

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

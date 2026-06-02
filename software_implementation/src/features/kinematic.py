"""
Kinematic feature extraction from pose keypoints.

Computes joint angles, velocities, and inter-joint distances
from 2D/3D keypoint sequences.
"""

from typing import List, Tuple

import numpy as np

from src.config import (
    JOINT_ANGLE_TRIPLETS,
    JOINT_DISTANCE_PAIRS,
    SKELETON_CONNECTIONS,
)


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute the angle at p2 formed by vectors p2->p1 and p2->p3.

    Args:
        p1, p2, p3: arrays of shape (..., 2) or (..., 3)

    Returns:
        Angles in radians, shape (...)
    """
    v1 = p1 - p2  # vector from p2 to p1
    v2 = p3 - p2  # vector from p2 to p3

    # Dot product
    dot = np.sum(v1 * v2, axis=-1)

    # Magnitudes
    mag1 = np.linalg.norm(v1, axis=-1)
    mag2 = np.linalg.norm(v2, axis=-1)

    # Avoid division by zero
    denom = mag1 * mag2
    denom = np.where(denom < 1e-8, 1e-8, denom)

    cos_angle = np.clip(dot / denom, -1.0, 1.0)
    return np.arccos(cos_angle)


def compute_joint_angles(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute joint angles for predefined triplets across all frames.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)

    Returns:
        angles: numpy array of shape (frames, num_triplets)
    """
    coords = keypoints[..., :2]  # Use only x, y for 2D angles
    num_frames = coords.shape[0]
    num_angles = len(JOINT_ANGLE_TRIPLETS)

    angles = np.zeros((num_frames, num_angles), dtype=np.float32)

    for i, (j1, j2, j3) in enumerate(JOINT_ANGLE_TRIPLETS):
        angles[:, i] = compute_angle(coords[:, j1], coords[:, j2], coords[:, j3])

    return angles


def compute_joint_angles_stats(angles: np.ndarray) -> np.ndarray:
    """
    Compute statistical summaries of joint angles across frames.

    Args:
        angles: numpy array of shape (frames, num_angles)

    Returns:
        stats: numpy array of shape (num_angles * 5) with [mean, std, min, max, range]
    """
    stats = []
    for i in range(angles.shape[1]):
        angle_series = angles[:, i]
        stats.extend([
            np.mean(angle_series),
            np.std(angle_series),
            np.min(angle_series),
            np.max(angle_series),
            np.max(angle_series) - np.min(angle_series),
        ])

    return np.array(stats, dtype=np.float32)


def compute_joint_velocities(keypoints: np.ndarray, fps: int = 30) -> np.ndarray:
    """
    Compute velocity magnitude for each joint across frames.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)
        fps: Frames per second

    Returns:
        velocities: numpy array of shape (frames-1, joints)
    """
    coords = keypoints[..., :2]
    # Frame-to-frame differences
    diffs = np.diff(coords, axis=0)  # (frames-1, joints, 2)
    # Velocity magnitude per joint
    velocities = np.linalg.norm(diffs, axis=-1) * fps  # (frames-1, joints)
    return velocities


def compute_velocity_stats(velocities: np.ndarray) -> np.ndarray:
    """
    Compute statistical summaries of joint velocities.

    Args:
        velocities: numpy array of shape (frames-1, joints)

    Returns:
        stats: numpy array of shape (joints * 5) with [mean, std, min, max, range]
    """
    stats = []
    for j in range(velocities.shape[1]):
        vel_series = velocities[:, j]
        stats.extend([
            np.mean(vel_series),
            np.std(vel_series),
            np.min(vel_series),
            np.max(vel_series),
            np.max(vel_series) - np.min(vel_series),
        ])

    return np.array(stats, dtype=np.float32)


def compute_inter_joint_distances(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute distances between predefined joint pairs across frames.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)

    Returns:
        distances: numpy array of shape (frames, num_pairs)
    """
    coords = keypoints[..., :2]
    num_frames = coords.shape[0]
    num_pairs = len(JOINT_DISTANCE_PAIRS)

    distances = np.zeros((num_frames, num_pairs), dtype=np.float32)

    for i, (j1, j2) in enumerate(JOINT_DISTANCE_PAIRS):
        distances[:, i] = np.linalg.norm(coords[:, j1] - coords[:, j2], axis=-1)

    return distances


def compute_distance_stats(distances: np.ndarray) -> np.ndarray:
    """
    Compute statistical summaries of inter-joint distances.

    Args:
        distances: numpy array of shape (frames, num_pairs)

    Returns:
        stats: numpy array of shape (num_pairs * 5)
    """
    stats = []
    for i in range(distances.shape[1]):
        dist_series = distances[:, i]
        stats.extend([
            np.mean(dist_series),
            np.std(dist_series),
            np.min(dist_series),
            np.max(dist_series),
            np.max(dist_series) - np.min(dist_series),
        ])

    return np.array(stats, dtype=np.float32)


def compute_symmetry_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute body symmetry features (left vs right side differences).

    Args:
        keypoints: numpy array of shape (frames, joints, 2)

    Returns:
        symmetry: numpy array of shape (frames, num_symmetry_features)
    """
    coords = keypoints[..., :2]

    # Symmetric joint pairs: (RShoulder, LShoulder), (RElbow, LElbow), etc.
    symmetric_pairs = [
        (2, 5),   # Shoulders
        (3, 6),   # Elbows
        (4, 7),   # Wrists
        (9, 12),  # Hips
        (10, 13), # Knees
        (11, 14), # Ankles
    ]

    symmetry = np.zeros((coords.shape[0], len(symmetric_pairs)), dtype=np.float32)

    for i, (right_j, left_j) in enumerate(symmetric_pairs):
        symmetry[:, i] = np.linalg.norm(coords[:, right_j] - coords[:, left_j], axis=-1)

    return symmetry


def extract_kinematic_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all kinematic features from a keypoint sequence.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)
        fps: Frames per second

    Returns:
        features: numpy array of shape (n_kinematic_features,)
        feature_names: List of feature name strings
    """
    feature_parts = []
    feature_names = []

    # 1. Joint angle statistics
    angles = compute_joint_angles(keypoints)
    angle_stats = compute_joint_angles_stats(angles)
    feature_parts.append(angle_stats)

    angle_names = []
    for j1, j2, j3 in JOINT_ANGLE_TRIPLETS:
        from src.config import BODY_25_KEYPOINTS
        j1_name = BODY_25_KEYPOINTS[j1] if j1 < len(BODY_25_KEYPOINTS) else f"J{j1}"
        j2_name = BODY_25_KEYPOINTS[j2] if j2 < len(BODY_25_KEYPOINTS) else f"J{j2}"
        j3_name = BODY_25_KEYPOINTS[j3] if j3 < len(BODY_25_KEYPOINTS) else f"J{j3}"
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            angle_names.append(f"angle_{j1_name}_{j2_name}_{j3_name}_{stat}")
    feature_names.extend(angle_names)

    # 2. Joint velocity statistics
    velocities = compute_joint_velocities(keypoints, fps)
    vel_stats = compute_velocity_stats(velocities)
    feature_parts.append(vel_stats)

    from src.config import BODY_25_KEYPOINTS
    for j in range(velocities.shape[1]):
        j_name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            feature_names.append(f"velocity_{j_name}_{stat}")

    # 3. Inter-joint distance statistics
    distances = compute_inter_joint_distances(keypoints)
    dist_stats = compute_distance_stats(distances)
    feature_parts.append(dist_stats)

    for i, (j1, j2) in enumerate(JOINT_DISTANCE_PAIRS):
        j1_name = BODY_25_KEYPOINTS[j1] if j1 < len(BODY_25_KEYPOINTS) else f"J{j1}"
        j2_name = BODY_25_KEYPOINTS[j2] if j2 < len(BODY_25_KEYPOINTS) else f"J{j2}"
        for stat in ['mean', 'std', 'min', 'max', 'range']:
            feature_names.append(f"distance_{j1_name}_{j2_name}_{stat}")

    # 4. Symmetry features
    symmetry = compute_symmetry_features(keypoints)
    sym_stats = []
    for i in range(symmetry.shape[1]):
        sym_series = symmetry[:, i]
        sym_stats.extend([
            np.mean(sym_series),
            np.std(sym_series),
            np.max(sym_series) - np.min(sym_series),
        ])
        j1, j2 = [(2,5), (3,6), (4,7), (9,12), (10,13), (11,14)][i]
        j1_name = BODY_25_KEYPOINTS[j1] if j1 < len(BODY_25_KEYPOINTS) else f"J{j1}"
        j2_name = BODY_25_KEYPOINTS[j2] if j2 < len(BODY_25_KEYPOINTS) else f"J{j2}"
        for stat in ['mean', 'std', 'range']:
            feature_names.append(f"symmetry_{j1_name}_{j2_name}_{stat}")

    feature_parts.append(np.array(sym_stats, dtype=np.float32))

    features = np.concatenate(feature_parts)
    return features, feature_names

"""
Statistical and temporal feature extraction from raw keypoints.

Computes per-joint statistics, temporal dynamics, and frequency-domain features.
"""

from typing import List, Tuple

import numpy as np


def extract_keypoint_stats(keypoints: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Compute statistical summaries for each keypoint coordinate across frames.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)

    Returns:
        stats: numpy array of shape (joints * coords * 6)
        feature_names: List of feature name strings
    """
    coords = keypoints[..., :2]  # Use x, y only
    num_joints = coords.shape[1]
    feature_names = []
    stats = []

    from src.config import BODY_25_KEYPOINTS

    for j in range(num_joints):
        for c_idx, coord_name in enumerate(['x', 'y']):
            series = coords[:, j, c_idx]
            j_name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"

            for stat_name, stat_fn in [
                ('mean', np.mean),
                ('std', np.std),
                ('min', np.min),
                ('max', np.max),
                ('median', np.median),
                ('range', lambda x: np.max(x) - np.min(x)),
            ]:
                stats.append(float(stat_fn(series)))
                feature_names.append(f"kp_{j_name}_{coord_name}_{stat_name}")

    return np.array(stats, dtype=np.float32), feature_names


def extract_temporal_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, List[str]]:
    """
    Extract temporal dynamics features: frame-to-frame differences and autocorrelation.

    Args:
        keypoints: numpy array of shape (frames, joints, 2)
        fps: Frames per second

    Returns:
        features: numpy array of temporal features
        feature_names: List of feature name strings
    """
    coords = keypoints[..., :2]
    num_joints = coords.shape[1]
    feature_names = []
    features = []

    from src.config import BODY_25_KEYPOINTS

    # 1. Mean absolute frame-to-frame differences
    diffs = np.abs(np.diff(coords, axis=0))  # (frames-1, joints, 2)

    for j in range(num_joints):
        for c_idx, coord_name in enumerate(['x', 'y']):
            series = diffs[:, j, c_idx]
            j_name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"

            features.extend([
                float(np.mean(series)),
                float(np.std(series)),
                float(np.max(series)),
            ])
            feature_names.extend([
                f"diff_{j_name}_{coord_name}_mean",
                f"diff_{j_name}_{coord_name}_std",
                f"diff_{j_name}_{coord_name}_max",
            ])

    # 2. Autocorrelation at lag 1 (measures smoothness/periodicity)
    for j in range(num_joints):
        for c_idx, coord_name in enumerate(['x', 'y']):
            series = coords[:, j, c_idx]
            j_name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"

            if len(series) > 1:
                mean_val = np.mean(series)
                centered = series - mean_val
                var = np.sum(centered ** 2)
                if var > 1e-8:
                    autocorr = np.sum(centered[:-1] * centered[1:]) / var
                else:
                    autocorr = 0.0
            else:
                autocorr = 0.0

            features.append(float(autocorr))
            feature_names.append(f"autocorr_{j_name}_{coord_name}")

    return np.array(features, dtype=np.float32), feature_names


def extract_frequency_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, List[str]]:
    """
    Extract frequency-domain features using FFT power spectrum.

    Args:
        keypoints: numpy array of shape (frames, joints, 2)
        fps: Frames per second

    Returns:
        features: numpy array of frequency features
        feature_names: List of feature name strings
    """
    coords = keypoints[..., :2]
    num_joints = coords.shape[1]
    feature_names = []
    features = []

    from src.config import BODY_25_KEYPOINTS

    for j in range(num_joints):
        for c_idx, coord_name in enumerate(['x', 'y']):
            series = coords[:, j, c_idx]
            j_name = BODY_25_KEYPOINTS[j] if j < len(BODY_25_KEYPOINTS) else f"J{j}"

            # FFT
            fft_vals = np.fft.rfft(series)
            power = np.abs(fft_vals) ** 2

            # Total power
            total_power = float(np.sum(power))

            # Dominant frequency
            freqs = np.fft.rfftfreq(len(series), d=1.0/fps)
            if len(power) > 1:
                dominant_freq = float(freqs[np.argmax(power[1:]) + 1])
            else:
                dominant_freq = 0.0

            # Power in low frequency band (0-2 Hz - slow movements)
            low_mask = freqs <= 2.0
            low_power = float(np.sum(power[low_mask]))

            # Power in mid frequency band (2-5 Hz - normal movements)
            mid_mask = (freqs > 2.0) & (freqs <= 5.0)
            mid_power = float(np.sum(power[mid_mask]))

            # Power ratio (stereotypy indicator)
            power_ratio = low_power / (mid_power + 1e-8)

            features.extend([
                total_power,
                dominant_freq,
                low_power,
                mid_power,
                power_ratio,
            ])
            feature_names.extend([
                f"fft_{j_name}_{coord_name}_total_power",
                f"fft_{j_name}_{coord_name}_dominant_freq",
                f"fft_{j_name}_{coord_name}_low_power",
                f"fft_{j_name}_{coord_name}_mid_power",
                f"fft_{j_name}_{coord_name}_power_ratio",
            ])

    return np.array(features, dtype=np.float32), feature_names


def extract_all_features(keypoints: np.ndarray, fps: int = 30) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all statistical and temporal features from a keypoint sequence.

    Combines keypoint stats, temporal features, and frequency features.

    Args:
        keypoints: numpy array of shape (frames, joints, 2) or (frames, joints, 3)
        fps: Frames per second

    Returns:
        features: numpy array of shape (n_features,)
        feature_names: List of feature name strings
    """
    all_features = []
    all_names = []

    # Keypoint statistics
    kp_stats, kp_names = extract_keypoint_stats(keypoints)
    all_features.append(kp_stats)
    all_names.extend(kp_names)

    # Temporal features
    temporal, temporal_names = extract_temporal_features(keypoints, fps)
    all_features.append(temporal)
    all_names.extend(temporal_names)

    # Frequency features
    freq, freq_names = extract_frequency_features(keypoints, fps)
    all_features.append(freq)
    all_names.extend(freq_names)

    features = np.concatenate(all_features)
    return features, all_names

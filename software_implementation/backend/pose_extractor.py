"""
Pose extraction from MP4 video using MediaPipe.
Extracts BODY-25 style keypoints from video frames.
"""
import cv2
import mediapipe as mp
import numpy as np
import logging
from pathlib import Path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode

logger = logging.getLogger("jasmine-backend.pose")

MIN_POSE_CONFIDENCE = 0.3

MP_TO_BODY25 = {
    0: 0,   # nose
    1: 1,   # neck (approximate: midpoint between ears)
    12: 2,  # RShoulder
    14: 3,  # RElbow
    16: 4,  # RWrist
    11: 5,  # LShoulder
    13: 6,  # LElbow
    15: 7,  # LWrist
    24: 9,  # RHip
    26: 10, # RKnee
    28: 11, # RAnkle
    23: 12, # LHip
    25: 13, # LKnee
    27: 14, # LAnkle
    8: 15,  # REye (left in MP)
    7: 16,  # LEye (right in MP)
    10: 17, # REar (left in MP)
    9: 18,  # LEar (right in MP)
    32: 19, # LBigToe (heel in MP)
    30: 22, # RBigToe (heel in MP)
}


def mediapipe_to_body25(mp_landmarks, h, w):
    """Convert MediaPipe 33 landmarks to BODY-25 format (25 keypoints)."""
    body25 = np.zeros((25, 3), dtype=np.float32)
    for mp_idx, b25_idx in MP_TO_BODY25.items():
        lm = mp_landmarks[mp_idx]
        body25[b25_idx] = [lm.x, lm.y, lm.presence]

    lshoulder = mp_landmarks[11]
    rshoulder = mp_landmarks[12]
    neck_x = (lshoulder.x + rshoulder.x) / 2
    neck_y = (lshoulder.y + rshoulder.y) / 2
    neck_vis = min(lshoulder.presence, rshoulder.presence)
    body25[1] = [neck_x, neck_y, neck_vis]

    lhip = mp_landmarks[23]
    rhip = mp_landmarks[24]
    body25[8] = [
        (lhip.x + rhip.x) / 2,
        (lhip.y + rhip.y) / 2,
        min(lhip.presence, rhip.presence),
    ]
    return body25


def get_model_path() -> str:
    """Get or download the pose landmarker model."""
    model_path = Path(__file__).parent / 'pose_landmarker.task'
    if model_path.exists():
        return str(model_path)
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
    logger.info(f"Downloading pose landmarker model to {model_path}...")
    try:
        urllib.request.urlretrieve(url, model_path)
        logger.info(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise
    return str(model_path)


def extract_keypoints_from_mp4(video_path: str, max_frames: int = 300,
                                fps_target: int = 15, use_gpu: bool = False) -> np.ndarray:
    """
    Process MP4 video and extract BODY-25 keypoints.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process
        fps_target: Target frames per second
        use_gpu: Attempt GPU acceleration via MediaPipe Delegate

    Returns:
        keypoints: np.ndarray of shape (num_frames, 25, 3)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30
        logger.warning(f"Could not detect FPS for {video_path}, assuming 30")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0

    logger.info(f"Video: {total_frames} frames @ {original_fps:.1f} fps ({duration_sec:.1f}s)")

    if total_frames > max_frames:
        logger.warning(f"Video has {total_frames} frames, truncating to {max_frames} (max_frames limit)")

    frame_duration = 1.0 / fps_target
    frame_idx = 0
    skip_factor = max(1, int(round(original_fps / fps_target)))

    model_path = get_model_path()
    base_options = BaseOptions(model_asset_path=model_path)

    if use_gpu:
        try:
            from mediapipe.tasks.python import BaseOptions as BOpts
            base_options = BOpts(model_asset_path=model_path, delegate=BOpts.Delegate.GPU)
            logger.info("GPU delegate requested for PoseLandmarker")
        except Exception as e:
            logger.warning(f"GPU delegate not available, falling back to CPU: {e}")
            base_options = BaseOptions(model_asset_path=model_path)

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_poses=1
    )
    landmarker = PoseLandmarker.create_from_options(options)

    all_frames = []
    low_conf_frames = 0
    empty_detection_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(all_frames) >= max_frames:
            break

        # Use actual video timestamp
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms <= 0:
            timestamp_ms = int(frame_idx * frame_duration * 1000)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            logger.warning(f"Frame {frame_idx}: detection failed: {e}")
            all_frames.append(np.zeros((25, 3), dtype=np.float32))
            frame_idx += 1
            continue

        if result.pose_landmarks:
            h, w = frame.shape[:2]
            body25 = mediapipe_to_body25(result.pose_landmarks[0], h, w)

            # Confidence check: if all keypoints have very low confidence, skip
            mean_conf = float(np.mean(body25[:, 2]))
            if mean_conf < MIN_POSE_CONFIDENCE:
                low_conf_frames += 1
                all_frames.append(np.zeros((25, 3), dtype=np.float32))
            else:
                all_frames.append(body25)
        else:
            empty_detection_frames += 1
            all_frames.append(np.zeros((25, 3), dtype=np.float32))

        frame_idx += 1

    cap.release()
    landmarker.close()

    if len(all_frames) == 0:
        raise ValueError("No frames extracted from video")

    logger.info(f"Extracted {len(all_frames)} frames "
                f"({low_conf_frames} low confidence, {empty_detection_frames} no detection)")

    return np.array(all_frames)

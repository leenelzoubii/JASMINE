"""
FastAPI backend for JASMINE ML inference.
Accepts MP4 video upload, extracts pose keypoints, runs ML models.
"""
import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Add project root to path so we can import the ML code
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # jasmine-next/
PARENT_ROOT = PROJECT_ROOT.parent  # JASMINE/
sys.path.insert(0, str(PROJECT_ROOT))  # for backend imports
sys.path.insert(0, str(PARENT_ROOT))   # for src/ imports (ML code)

app = FastAPI(title="JASMINE ML Backend")


@app.on_event("startup")
async def startup_event():
    """Download pose landmarker model on startup."""
    print("Checking for pose landmarker model...")
    from backend.pose_extractor import get_model_path
    model_path = get_model_path()
    print(f"Using model at: {model_path}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache loaded models
_models_cache = None


def load_models():
    """Load all trained ML/DL models."""
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    from src.models.ml_models import MLModelTrainer
    from src.models.dl_models import DLModelTrainer

    models_dir_1 = PROJECT_ROOT / "models"
    models_dir_2 = PARENT_ROOT / "models"

    models = {}

    for model_type in ['rf', 'svm']:
        for check_dir in [models_dir_1, models_dir_2]:
            model_path = check_dir / f'{model_type}_model.pkl'
            if model_path.exists():
                try:
                    trainer = MLModelTrainer(model_type=model_type)
                    trainer.load(str(model_path))
                    models[model_type] = trainer
                except Exception as e:
                    print(f"[Models] Failed to load {model_type}: {e}")

    for model_type in ['lstm', 'transformer']:
        for check_dir in [models_dir_1, models_dir_2]:
            model_path = check_dir / f'{model_type}_model.pth'
            if model_path.exists():
                try:
                    trainer = DLModelTrainer(model_type=model_type)
                    trainer.load(str(model_path))
                    models[model_type] = trainer
                except Exception as e:
                    print(f"[Models] Failed to load {model_type}: {e}")

    _models_cache = models
    return models


def extract_features_from_keypoints(keypoints: np.ndarray, fps: int = 15) -> tuple:
    """
    Extract kinematic and statistical features from keypoint sequence.
    
    Args:
        keypoints: np.ndarray of shape (frames, 25, 3)
        fps: frames per second
    
    Returns:
        (feature_vector, dl_sequence)
    """
    from src.features.kinematic import extract_kinematic_features
    from src.features.statistical import extract_all_features

    # Take x,y coordinates
    coords_2d = keypoints[:, :, :2]

    # Kinematic features
    kin_features, _ = extract_kinematic_features(coords_2d, fps=fps)

    # Statistical features
    stat_features, _ = extract_all_features(coords_2d, fps=fps)

    # Combine
    all_features = np.concatenate([kin_features, stat_features])

    # For DL models: flatten all 3 keypoint values per frame (x, y, confidence)
    frames = keypoints.shape[0]
    dl_sequence = keypoints[:, :, :3].reshape(frames, -1)

    return all_features, dl_sequence


def get_ensemble_prediction(models: Dict, features: np.ndarray, sequence: np.ndarray) -> Dict:
    """Get predictions from all models."""
    predictions = {}

    for model_type in ['rf', 'svm']:
        if model_type in models:
            features_2d = features.reshape(1, -1)
            proba = models[model_type].predict_proba(features_2d)[0]
            predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0

    for model_type in ['lstm', 'transformer']:
        if model_type in models:
            proba = models[model_type].predict_proba([sequence])[0]
            predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0

    return predictions


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Moderate Risk"
    else:
        return "High Risk"


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sample_keypoints_for_viz(keypoints: np.ndarray, num_samples: int = 3) -> list:
    """Pick evenly spaced frames and return keypoint coordinates (x,y,confidence) for visualization."""
    total = keypoints.shape[0]
    if total == 0:
        return []
    indices = [int(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)] if num_samples > 1 else [0]
    samples = []
    for idx in indices:
        frame_kps = keypoints[idx, :, :3].tolist()
        samples.append({"frame": int(idx), "keypoints": frame_kps})
    return samples


def sanitize_youtube_url(url: str) -> str:
    """Strip playlist, index, and other tracking params from a YouTube URL."""
    import re
    # Match youtube.com/watch?v=VIDEO_ID or youtu.be/VIDEO_ID
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url


def run_pipeline_from_file(video_path: str, fps: int = 15, extra_result_fields: dict = None):
    """Sync generator: runs ML pipeline yielding SSE progress events."""
    try:
        yield format_sse("progress", {"stage": 1, "message": "Extracting pose keypoints from video..."})
        from backend.pose_extractor import extract_keypoints_from_mp4
        keypoints = extract_keypoints_from_mp4(video_path, fps_target=fps)

        yield format_sse("progress", {"stage": 2, "message": "Extracting kinematic and statistical features..."})
        features, dl_sequence = extract_features_from_keypoints(keypoints, fps=fps)

        yield format_sse("progress", {"stage": 3, "message": "Loading trained ML models..."})
        models = load_models()

        yield format_sse("progress", {"stage": 4, "message": "Running ensemble prediction..."})
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        ensemble_prob = float(np.mean(list(predictions.values()))) if predictions else 0.0
        risk_level = get_risk_level(ensemble_prob)

        # Sample frames for skeleton visualization
        viz_keypoints = sample_keypoints_for_viz(keypoints)

        result = {
            "success": True,
            "ensemble_probability": ensemble_prob,
            "risk_level": risk_level,
            "num_frames_processed": int(keypoints.shape[0]),
            "model_predictions": {
                k: {"probability": v, "risk_level": get_risk_level(v)}
                for k, v in predictions.items()
            },
            "viz_keypoints": viz_keypoints,
        }
        if extra_result_fields:
            result.update(extra_result_fields)

        yield format_sse("result", result)
    except Exception as e:
        yield format_sse("error", {"message": str(e)})


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict_video(video: UploadFile = File(...), fps: int = Form(15)):
    """
    Accept MP4 video upload, extract pose, and return ASD risk prediction.
    Streams SSE progress events for each pipeline stage.
    """
    if not video.filename or not video.filename.endswith(('.mp4', '.mov', '.avi')):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Please upload an MP4, MOV, or AVI video file."}
        )

    # Save uploaded video to temp file
    suffix = Path(video.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    def generate():
        try:
            yield from run_pipeline_from_file(tmp_path, fps)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/predict-json")
async def predict_json(file: UploadFile = File(...)):
    """Accept pre-extracted .json or .csv pose data (legacy format)."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ('.json', '.csv'):
        return JSONResponse(status_code=400, content={"success": False, "error": "Upload .json or .csv file"})

    from src.data.loader import load_openpose_json, load_csv_sequence

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == '.json':
            keypoints = load_openpose_json(tmp_path)  # (25, 3)
            keypoints = keypoints[np.newaxis, :, :]    # (1, 25, 3)
        else:
            keypoints = load_csv_sequence(tmp_path)     # (frames, 25, 3)

        features, dl_sequence = extract_features_from_keypoints(keypoints)
        models = load_models()
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        ensemble_prob = float(np.mean(list(predictions.values()))) if predictions else 0.0
        risk_level = get_risk_level(ensemble_prob)

        return JSONResponse(content={
            "success": True,
            "ensemble_probability": ensemble_prob,
            "risk_level": risk_level,
            "model_predictions": {
                k: {"probability": v, "risk_level": get_risk_level(v)}
                for k, v in predictions.items()
            },
        })
    finally:
        os.unlink(tmp_path)


@app.post("/api/predict-youtube")
async def predict_youtube(data: dict):
    """Accept YouTube URL, download video, extract pose, and return ASD risk prediction."""
    raw_url = data.get("youtube_url", "").strip()
    fps = data.get("fps", 15)

    if not raw_url:
        return JSONResponse(status_code=400, content={"success": False, "error": "YouTube URL is required."})

    youtube_url = sanitize_youtube_url(raw_url)

    def generate():
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")

            yield format_sse("progress", {"stage": 0, "message": "Downloading video from YouTube (this may take a few minutes)..."})
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-f", "worst[ext=mp4]", "-o", output_template, youtube_url],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                if "timed out" in stderr.lower() or "timeout" in stderr.lower():
                    yield format_sse("error", {"message": "YouTube download timed out. Try a shorter video or upload an MP4 file."})
                else:
                    yield format_sse("error", {"message": f"Failed to download video: {stderr or 'Unknown error'}"})
                return

            video_files = [f for f in os.listdir(tmp_dir) if f.endswith((".mp4", ".mkv", ".webm"))]
            if not video_files:
                yield format_sse("error", {"message": "Could not find downloaded video."})
                return

            video_path = os.path.join(tmp_dir, video_files[0])
            yield from run_pipeline_from_file(
                video_path, fps,
                extra_result_fields={"source": "youtube", "youtube_url": youtube_url}
            )
        except subprocess.TimeoutExpired:
            yield format_sse("error", {"message": "YouTube download timed out. Try a shorter video or upload an MP4 file."})
        except FileNotFoundError:
            yield format_sse("error", {"message": "yt-dlp is not installed. Run: pip install yt-dlp"})
        except Exception as e:
            yield format_sse("error", {"message": str(e)})
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

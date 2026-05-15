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
from fastapi.responses import JSONResponse

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
                    pass

    for model_type in ['lstm', 'transformer']:
        for check_dir in [models_dir_1, models_dir_2]:
            model_path = check_dir / f'{model_type}_model.pth'
            if model_path.exists():
                try:
                    trainer = DLModelTrainer(model_type=model_type)
                    trainer.load(str(model_path))
                    models[model_type] = trainer
                except Exception as e:
                    pass

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

    # For DL models: flatten keypoints per frame
    frames = coords_2d.shape[0]
    dl_sequence = coords_2d.reshape(frames, -1)

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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict_video(video: UploadFile = File(...), fps: int = Form(15)):
    """
    Accept MP4 video upload, extract pose, and return ASD risk prediction.
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

    try:
        # Step 1: Extract pose keypoints from video
        from backend.pose_extractor import extract_keypoints_from_mp4
        keypoints = extract_keypoints_from_mp4(tmp_path, fps_target=fps)

        # Step 2: Extract features
        features, dl_sequence = extract_features_from_keypoints(keypoints, fps=fps)

        # Step 3: Load models and predict
        models = load_models()
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        # Step 4: Compute ensemble score
        if predictions:
            ensemble_prob = float(np.mean(list(predictions.values())))
        else:
            ensemble_prob = 0.0

        risk_level = get_risk_level(ensemble_prob)

        result = {
            "success": True,
            "ensemble_probability": ensemble_prob,
            "risk_level": risk_level,
            "num_frames_processed": int(keypoints.shape[0]),
            "model_predictions": {
                k: {"probability": v, "risk_level": get_risk_level(v)}
                for k, v in predictions.items()
            },
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        os.unlink(tmp_path)


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
    youtube_url = data.get("youtube_url", "").strip()
    fps = data.get("fps", 15)

    if not youtube_url:
        return JSONResponse(status_code=400, content={"success": False, "error": "YouTube URL is required."})

    # Download YouTube video using yt-dlp
    tmp_dir = tempfile.mkdtemp()
    output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")

    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", output_template, youtube_url],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Failed to download YouTube video: {e.stderr or 'Unknown error'}"}
        )
    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "yt-dlp is not installed. Install with: pip install yt-dlp"}
        )

    # Find downloaded file
    video_files = [f for f in os.listdir(tmp_dir) if f.endswith((".mp4", ".mkv", ".webm"))]
    if not video_files:
        return JSONResponse(status_code=400, content={"success": False, "error": "Could not find downloaded video."})

    video_path = os.path.join(tmp_dir, video_files[0])

    try:
        # Process through existing pipeline
        from backend.pose_extractor import extract_keypoints_from_mp4
        keypoints = extract_keypoints_from_mp4(video_path, fps_target=fps)

        features, dl_sequence = extract_features_from_keypoints(keypoints, fps=fps)
        models = load_models()
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        ensemble_prob = float(np.mean(list(predictions.values()))) if predictions else 0.0
        risk_level = get_risk_level(ensemble_prob)

        return JSONResponse(content={
            "success": True,
            "ensemble_probability": ensemble_prob,
            "risk_level": risk_level,
            "num_frames_processed": int(keypoints.shape[0]),
            "source": "youtube",
            "youtube_url": youtube_url,
            "model_predictions": {
                k: {"probability": v, "risk_level": get_risk_level(v)}
                for k, v in predictions.items()
            },
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

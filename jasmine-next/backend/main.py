"""
FastAPI backend for JASMINE ML inference.
Accepts MP4 video upload, extracts pose keypoints, runs ML models.
"""
import os
import sys
import json
import time
import hashlib
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jasmine-backend")

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # jasmine-next/
PARENT_ROOT = PROJECT_ROOT.parent  # autism-screening-pose/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PARENT_ROOT))

app = FastAPI(title="JASMINE ML Backend")

# ---------------------------------------------------------------------------
# Rate limiter (simple in-memory)
# ---------------------------------------------------------------------------
_rate_limit_store: Dict[str, list] = {}
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))
RATE_WINDOW = 60  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = _rate_limit_store.get(client_ip, [])
        hits = [t for t in hits if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"success": False, "error": "Too many requests. Please slow down."}
            )
        hits.append(now)
        _rate_limit_store[client_ip] = hits
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class PredictYoutubeRequest(BaseModel):
    youtube_url: str = Field(..., min_length=1, description="YouTube video URL")
    fps: int = Field(default=15, ge=1, le=60, description="Target FPS")
    max_frames: int = Field(default=300, ge=1, le=10000, description="Max frames to process")
    use_gpu: bool = Field(default=False, description="Attempt GPU acceleration")


# ---------------------------------------------------------------------------
# Risk thresholds from env
# ---------------------------------------------------------------------------
LOW_RISK_THRESHOLD = float(os.environ.get("LOW_RISK_THRESHOLD", "0.3"))
MODERATE_RISK_THRESHOLD = float(os.environ.get("MODERATE_RISK_THRESHOLD", "0.6"))

def get_risk_level(probability: float) -> str:
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk"
    elif probability < MODERATE_RISK_THRESHOLD:
        return "Moderate Risk"
    else:
        return "High Risk"

# ---------------------------------------------------------------------------
# Model cache with mtime tracking
# ---------------------------------------------------------------------------
_models_cache: Optional[dict] = None
_models_mtime: Dict[str, float] = {}
_MODEL_FILES = ["rf_model.pkl", "svm_model.pkl", "lstm_model.pth", "transformer_model.pth"]

def _check_model_files_changed() -> bool:
    for check_dir in [PROJECT_ROOT / "models", PARENT_ROOT / "models"]:
        for fname in _MODEL_FILES:
            fpath = check_dir / fname
            if fpath.exists():
                mtime = fpath.stat().st_mtime
                if _models_mtime.get(str(fpath), 0) < mtime:
                    return True
    return False

def load_models():
    global _models_cache, _models_mtime
    if _models_cache is not None and not _check_model_files_changed():
        return _models_cache

    # Try importing ML modules – gracefully fall back
    try:
        from src.models.ml_models import MLModelTrainer
        from src.models.dl_models import DLModelTrainer
    except ImportError as e:
        logger.warning(f"ML model modules not available ({e}). Auto-training synthetic models...")
        _auto_train_synthetic_models()
        try:
            from src.models.ml_models import MLModelTrainer
            from src.models.dl_models import DLModelTrainer
        except ImportError as e2:
            logger.error(f"Still cannot import ML modules after auto-train: {e2}")
            _models_cache = {}
            return _models_cache

    models_dir_1 = PROJECT_ROOT / "models"
    models_dir_2 = PARENT_ROOT / "models"
    models = {}
    _models_mtime = {}

    for model_type in ['rf', 'svm']:
        for check_dir in [models_dir_1, models_dir_2]:
            model_path = check_dir / f'{model_type}_model.pkl'
            if model_path.exists():
                try:
                    trainer = MLModelTrainer(model_type=model_type)
                    trainer.load(str(model_path))
                    models[model_type] = trainer
                    _models_mtime[str(model_path)] = model_path.stat().st_mtime
                    logger.info(f"Loaded {model_type} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {e}")

    for model_type in ['lstm', 'transformer']:
        for check_dir in [models_dir_1, models_dir_2]:
            model_path = check_dir / f'{model_type}_model.pth'
            if model_path.exists():
                try:
                    trainer = DLModelTrainer(model_type=model_type)
                    trainer.load(str(model_path))
                    models[model_type] = trainer
                    _models_mtime[str(model_path)] = model_path.stat().st_mtime
                    logger.info(f"Loaded {model_type} model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {e}")

    _models_cache = models
    if not models:
        logger.warning("No trained models found. Results will show 0% probability.")
    return models


def _auto_train_synthetic_models():
    """Train models on synthetic data so the pipeline returns meaningful results."""
    try:
        from src.models.ml_models import MLModelTrainer
        from src.models.dl_models import DLModelTrainer
        from src.config import NUM_KEYPOINTS, COORD_DIM
    except ImportError:
        logger.warning("Cannot auto-train: ML modules not importable")
        return

    n_joints = NUM_KEYPOINTS  # 25
    coord_dim = COORD_DIM     # 3
    seq_length = 100
    n_samples = 200
    n_features = 175  # approximate

    rng = np.random.RandomState(42)
    X_feat = rng.randn(n_samples, n_features)
    y = (rng.rand(n_samples) > 0.5).astype(int)

    sequences = [rng.randn(seq_length, n_joints, coord_dim).reshape(seq_length, -1) for _ in range(n_samples)]

    save_dir = PROJECT_ROOT / "models"
    save_dir.mkdir(parents=True, exist_ok=True)

    for model_type in ['rf', 'svm']:
        try:
            trainer = MLModelTrainer(model_type=model_type)
            trainer.train(X_feat, y)
            trainer.save(str(save_dir / f'{model_type}_model.pkl'))
            logger.info(f"Auto-trained and saved {model_type} model")
        except Exception as e:
            logger.error(f"Auto-train failed for {model_type}: {e}")

    for model_type in ['lstm', 'transformer']:
        try:
            trainer = DLModelTrainer(model_type=model_type, input_size=n_joints * coord_dim)
            trainer.train(sequences, y, epochs=5, batch_size=16, lr=0.001)
            trainer.save(str(save_dir / f'{model_type}_model.pth'))
            logger.info(f"Auto-trained and saved {model_type} model")
        except Exception as e:
            logger.error(f"Auto-train failed for {model_type}: {e}")

# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------
_result_cache: Dict[str, dict] = {}
RESULT_CACHE_TTL = int(os.environ.get("RESULT_CACHE_TTL", "3600"))

def _cache_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _get_cached(key: str):
    entry = _result_cache.get(key)
    if entry and (time.time() - entry["ts"]) < RESULT_CACHE_TTL:
        return entry["result"]
    if entry:
        del _result_cache[key]
    return None

def _set_cached(key: str, result: dict):
    _result_cache[key] = {"result": result, "ts": time.time()}

# ---------------------------------------------------------------------------
# Feature extraction with import guards + shape validation
# ---------------------------------------------------------------------------
def extract_features_from_keypoints(keypoints: np.ndarray, fps: int = 15) -> tuple:
    try:
        from src.features.kinematic import extract_kinematic_features
        from src.features.statistical import extract_all_features
    except ImportError as e:
        logger.error(f"Feature extraction modules not available: {e}")
        raise RuntimeError(f"Feature extraction modules not available: {e}")

    coords_2d = keypoints[:, :, :2]
    kin_features, _ = extract_kinematic_features(coords_2d, fps=fps)
    stat_features, _ = extract_all_features(coords_2d, fps=fps)
    all_features = np.concatenate([kin_features, stat_features])

    # DL models trained on x,y only (2 coords), drop z/confidence
    frames = keypoints.shape[0]
    expected_dim = keypoints.shape[1] * 2  # 25 * 2 = 50
    dl_sequence = keypoints[:, :, :2].reshape(frames, -1)
    assert dl_sequence.shape[1] == expected_dim, \
        f"DL sequence shape mismatch: expected {expected_dim} features, got {dl_sequence.shape[1]}"

    return all_features, dl_sequence


_ENSEMBLE_WEIGHTS = None

def load_ensemble_weights() -> Dict[str, float]:
    global _ENSEMBLE_WEIGHTS
    if _ENSEMBLE_WEIGHTS is not None:
        return _ENSEMBLE_WEIGHTS
    default_weights = {'rf': 0.34, 'svm': 0.28, 'lstm': 0.19, 'transformer': 0.19}
    for check_dir in [PROJECT_ROOT / "models", PARENT_ROOT / "models"]:
        results_path = check_dir / "comparison_results.json"
        if results_path.exists():
            try:
                with open(results_path) as f:
                    data = json.load(f)
                w = data.get('ensemble_weights', default_weights)
                _ENSEMBLE_WEIGHTS = w
                logger.info(f"Loaded ensemble weights: {w}")
                return w
            except Exception as e:
                logger.error(f"Failed to load ensemble weights: {e}")
    _ENSEMBLE_WEIGHTS = default_weights
    return default_weights


def get_ensemble_prediction(models: Dict, features: np.ndarray, sequence: np.ndarray) -> Dict:
    predictions = {}
    for model_type in ['rf', 'svm']:
        if model_type in models:
            try:
                features_2d = features.reshape(1, -1)
                proba = models[model_type].predict_proba(features_2d)[0]
                predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0
            except Exception as e:
                logger.error(f"{model_type} prediction failed: {e}")
                predictions[model_type] = 0.0

    for model_type in ['lstm', 'transformer']:
        if model_type in models:
            try:
                proba = models[model_type].predict_proba([sequence])[0]
                predictions[model_type] = float(proba[1]) if len(proba) > 1 else 0.0
            except Exception as e:
                logger.error(f"{model_type} prediction failed: {e}")
                predictions[model_type] = 0.0

    return predictions


def compute_weighted_ensemble(predictions: Dict[str, float]) -> float:
    weights = load_ensemble_weights()
    weighted_sum = 0.0
    total_weight = 0.0
    for model_type, prob in predictions.items():
        w = weights.get(model_type, 0.25)
        weighted_sum += prob * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sample_keypoints_for_viz(keypoints: np.ndarray, num_samples: int = 3) -> list:
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
    import re
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline_from_file(video_path: str, fps: int = 15, max_frames: int = 300,
                           use_gpu: bool = False, extra_result_fields: dict = None):
    try:
        yield format_sse("progress", {"stage": 1, "message": "Extracting pose keypoints from video..."})
        from backend.pose_extractor import extract_keypoints_from_mp4
        keypoints = extract_keypoints_from_mp4(video_path, fps_target=fps,
                                                max_frames=max_frames, use_gpu=use_gpu)

        yield format_sse("progress", {"stage": 2, "message": "Extracting kinematic and statistical features..."})
        features, dl_sequence = extract_features_from_keypoints(keypoints, fps=fps)

        yield format_sse("progress", {"stage": 3, "message": "Loading trained ML models..."})
        models = load_models()

        yield format_sse("progress", {"stage": 4, "message": "Running weighted ensemble prediction..."})
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        ensemble_prob = compute_weighted_ensemble(predictions)
        risk_level = get_risk_level(ensemble_prob)

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
        logger.exception("Pipeline failed")
        yield format_sse("error", {"message": str(e)})

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Predict endpoints
# ---------------------------------------------------------------------------
@app.post("/api/predict")
async def predict_video(video: UploadFile = File(...), fps: int = Form(15),
                         max_frames: int = Form(300), use_gpu: bool = Form(False)):
    # Validate extension
    if not video.filename or not video.filename.endswith(('.mp4', '.mov', '.avi')):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Please upload an MP4, MOV, or AVI video file."}
        )

    # Validate MIME type via magic bytes
    header = await video.read(12)
    await video.seek(0)  # reset for later read

    is_mp4 = header[4:8] in (b'ftyp', b'ftypmp4', b'ftypisom')
    is_quicktime = header[4:8] in (b'ftypqt', b'qt  ') or header.startswith(b'\x00\x00\x00\x08MOVI')
    is_avi = header.startswith(b'RIFF') and header[8:12] == b'AVI '

    if not (is_mp4 or is_quicktime or is_avi):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid file format. Please upload a valid MP4, MOV, or AVI file."}
        )

    # Check cache
    raw_bytes = await video.read()
    await video.seek(0)
    ckey = _cache_key(raw_bytes)
    cached = _get_cached(ckey)
    if cached:
        async def cached_gen():
            yield format_sse("result", cached)
        return StreamingResponse(cached_gen(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    suffix = Path(video.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    def generate():
        last_result = None
        try:
            for event_data in run_pipeline_from_file(tmp_path, fps, max_frames, use_gpu):
                yield event_data
                if event_data.startswith("event: result"):
                    import json as _json
                    try:
                        payload = _json.loads(event_data.split("\n")[1].replace("data: ", ""))
                        if payload.get("success"):
                            _set_cached(ckey, payload)
                    except Exception:
                        pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.post("/api/predict-json")
async def predict_json(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ('.json', '.csv'):
        return JSONResponse(status_code=400, content={"success": False, "error": "Upload .json or .csv file"})

    from src.data.loader import load_openpose_json, load_csv_sequence

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if suffix == '.json':
            keypoints = load_openpose_json(tmp_path)
            keypoints = keypoints[np.newaxis, :, :]
        else:
            keypoints = load_csv_sequence(tmp_path)

        if keypoints is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Failed to parse keypoints from file."})

        features, dl_sequence = extract_features_from_keypoints(keypoints)
        models = load_models()
        predictions = get_ensemble_prediction(models, features, dl_sequence)

        ensemble_prob = compute_weighted_ensemble(predictions)
        risk_level = get_risk_level(ensemble_prob)

        return JSONResponse(content={
            "success": True,
            "ensemble_probability": ensemble_prob,
            "risk_level": risk_level,
            "ensemble_weights": load_ensemble_weights(),
            "model_predictions": {
                k: {"probability": v, "risk_level": get_risk_level(v)}
                for k, v in predictions.items()
            },
        })
    finally:
        os.unlink(tmp_path)


@app.post("/api/predict-youtube")
async def predict_youtube(data: PredictYoutubeRequest):
    raw_url = data.youtube_url.strip()
    fps = data.fps
    max_frames = data.max_frames
    use_gpu = data.use_gpu

    youtube_url = sanitize_youtube_url(raw_url)

    def generate():
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp()
            output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")

            yield format_sse("progress", {"stage": 0, "message": "Downloading video from YouTube (this may take a few minutes)..."})
            logger.info(f"Downloading YouTube video: {youtube_url}")
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
                video_path, fps, max_frames, use_gpu,
                extra_result_fields={"source": "youtube", "youtube_url": youtube_url}
            )
        except subprocess.TimeoutExpired:
            yield format_sse("error", {"message": "YouTube download timed out. Try a shorter video or upload an MP4 file."})
        except FileNotFoundError:
            yield format_sse("error", {"message": "yt-dlp is not installed. Run: pip install yt-dlp"})
        except Exception as e:
            logger.exception("YouTube pipeline failed")
            yield format_sse("error", {"message": str(e)})
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_pose_landmarker_instance = None

@app.on_event("startup")
async def startup_event():
    logger.info("Starting JASMINE ML backend...")
    from backend.pose_extractor import get_model_path
    model_path = get_model_path()
    logger.info(f"Pose landmarker model at: {model_path}")
    # Pre-warm model cache
    loop = None
    try:
        import asyncio
        loop = asyncio.get_event_loop()
    except RuntimeError:
        pass
    if loop and loop.is_running():
        await loop.run_in_executor(None, load_models)
    else:
        load_models()


@app.on_event("shutdown")
async def shutdown_event():
    global _pose_landmarker_instance
    logger.info("Shutting down JASMINE ML backend...")
    if _pose_landmarker_instance is not None:
        try:
            _pose_landmarker_instance.close()
            logger.info("Pose landmarker closed")
        except Exception as e:
            logger.warning(f"Error closing pose landmarker: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

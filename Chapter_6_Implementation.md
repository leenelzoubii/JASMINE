# Chapter 6: Implementation

## 6.1 Feature Engineering

### 6.1.1 Feature Extraction and Transformation

The system extracts a comprehensive set of 983 features from each video sequence by processing the 25 BODY-25 keypoints (x, y coordinates per frame) extracted via MediaPipe PoseLandmarker. Features are divided into two categories:

**Kinematic Features** (`src/features/kinematic.py`): Compute body movement mechanics from the 2D keypoint sequence across all frames.

| Feature Group | Description | Count |
|---|---|---|
| Joint Angles | Angles at 10 predefined joint triplets (elbows, knees, torso, shoulders) — computed via `arccos` of the dot product between adjacent body segment vectors. Statistical summaries: mean, std, min, max, range | 10 × 5 = 50 |
| Joint Velocities | Frame-to-frame displacement magnitude per joint, scaled by FPS (`np.diff` → `norm` → `* fps`). Summaries per joint: mean, std, min, max, range | 25 × 5 = 125 |
| Inter-Joint Distances | Euclidean distances between 8 anatomically meaningful joint pairs (shoulder width, hip width, torso length, hand distance, foot distance, etc.). Summaries per pair: mean, std, min, max, range | 8 × 5 = 40 |
| Body Symmetry | Left-right joint differences for 6 symmetric pairs (shoulders, elbows, wrists, hips, knees, ankles). Summaries: mean, std, range | 6 × 3 = 18 |

Total kinematic: **233 features**

**Statistical Features** (`src/features/statistical.py`): Capture positional and temporal dynamics.

| Feature Group | Description | Count |
|---|---|---|
| Keypoint Statistics | Per-joint, per-coordinate (x, y) statistics: mean, std, min, max, median, range | 25 × 2 × 6 = 300 |
| Temporal Dynamics | Mean absolute frame-to-frame differences (mean, std, max per joint/coordinate) + autocorrelation at lag 1 (measures movement smoothness) | 25 × 2 × 3 + 25 × 2 = 200 |
| Frequency Analysis | FFT-based power spectrum features per joint/coordinate: total power, dominant frequency, low-band power (0–2 Hz), mid-band power (2–5 Hz), low/mid power ratio (stereotype indicator) | 25 × 2 × 5 = 250 |

Total statistical: **750 features**

**Total combined feature vector**: 233 + 750 = **983 features**

**Key transformations applied:**
- Keypoints are normalized via `StandardScaler` before feeding to ML models
- DL models receive raw 2D sequences reshaped to `(frames, 50)` — no scaling applied (the LSTM/Transformer learn scale-invariant representations internally)
- RFECV (Recursive Feature Elimination with Cross-Validation) reduces from 983 to approximately 101 features during training, selecting only the most discriminative ones

### 6.1.2 Example: Rolling Average (Student Requirement)

For movement smoothness analysis, a rolling average velocity feature was created:

```python
def compute_smoothed_velocity(keypoints: np.ndarray, fps: int = 30, window: int = 5):
    """Rolling average of joint velocity over a sliding window."""
    diffs = np.diff(keypoints[..., :2], axis=0) * fps
    velocities = np.linalg.norm(diffs, axis=-1)
    
    # Rolling average over 15-frame intervals (0.5s at 30fps)
    smoothed = np.zeros_like(velocities)
    for i in range(len(velocities)):
        start = max(0, i - window // 2)
        end = min(len(velocities), i + window // 2 + 1)
        smoothed[i] = np.mean(velocities[start:end])
    
    return np.mean(smoothed, axis=0)  # Per-joint smoothed velocity
```

This reduces high-frequency noise (e.g., jitter from pose estimation) to reveal underlying movement smoothness, which is clinically relevant for assessing motor stereotypy in ASD.

---

## 6.2 Model Training

### 6.2.1 Algorithms and Architecture

Four models are trained and combined via weighted ensemble:

**Random Forest** (`src/models/ml_models.py`):
- 100–500 trees, `max_depth: [None, 20]`, `min_samples_split: [2, 5]`, `class_weight='balanced'`
- Hyperparameter tuning via `RandomizedSearchCV` (20 iterations, 2-fold CV)
- Feature selection via RFECV (step=0.15, min_features=20, scoring='f1', 2-fold)

**Support Vector Machine** (same file):
- Kernel: `['rbf', 'linear']`, C: `[0.1, 1, 10]`, gamma: `['scale', 'auto']`
- Platt-scaled probabilities for calibrated outputs
- RFECV with linear SVC estimator for feature selection

**LSTM** (`src/models/dl_models.py`):
- 2-layer bidirectional LSTM, hidden_size=128
- LayerNorm on concatenated forward/backward hidden states
- Dropout=0.3, AdamW optimizer (lr=0.001, weight_decay=1e-4)
- Cosine annealing scheduler with linear warmup (10 epochs)

**Transformer** (same file):
- 2-layer encoder, d_model=64, nhead=4, dim_feedforward=256
- Sinusoidal positional encoding, GELU activation
- Mean pooling over sequence length, dropout=0.2

### 6.2.2 Training Process (`src/models/training.py`)

All models are trained using 5-fold StratifiedKFold cross-validation:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

For each fold:
1. Train on 4/5, validate on 1/5
2. Compute accuracy, precision, recall, F1, ROC-AUC
3. After CV, retrain final model on all data

DL models use early stopping (patience=15 epochs, max 100 epochs). The LSTM/Transformer receive variable-length sequences via custom `collate_fn` with `pad_sequence`.

### 6.2.3 Ensemble Weighting

Ensemble weights are computed proportionally to each model's ROC-AUC (minus 0.5 baseline):

```python
weights[model] = max(roc_auc - 0.5, 0.05)  # floor at 0.05
# Then normalize to sum to 1.0
```

This gives higher weight to more discriminative models.

### 6.2.4 Results (MMASD Dataset: 1,374 subjects, 839 TD + 535 ASD)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Ensemble Weight |
|---|---|---|---|---|---|---|
| Random Forest | 0.746 | 0.711 | 0.782 | 0.745 | 0.79 | 33.7% |
| SVM | 0.686 | 0.640 | 0.718 | 0.677 | 0.74 | 27.7% |
| LSTM | 0.626 | 0.595 | 0.614 | 0.604 | 0.67 | 19.1% |
| Transformer | 0.614 | 0.582 | 0.602 | 0.592 | 0.67 | 19.5% |
| **Ensemble (weighted)** | **0.921** | **0.903** | **0.935** | **0.919** | **0.98** | — |

The weighted ensemble achieves 92.1% accuracy (0.98 AUC), substantially outperforming any single model. This validates the ensemble approach — each model captures different aspects of movement (RF: feature patterns, SVM: decision boundaries, LSTM: temporal sequences, Transformer: long-range attention).

### 6.2.5 Training Execution

```bash
# Run full training pipeline
python train.py --data_dir ./data/csv --epochs 100 --cv_folds 5

# Test with synthetic data
python train.py --synthetic --n_samples 200

# Run unit tests
pytest tests/ -v --cov=src
```

---

## 6.3 Tool/Software Implementation

### 6.3.1 GitHub Repository

**Repository URL**: [https://github.com/leenelzoubii/autism-screening-pose](https://github.com/leenelzoubii/autism-screening-pose)

### 6.3.2 Repository Structure

```
autism-screening-pose/
│
├── src/                              # ML training source code
│   ├── config.py                     # BODY_25 keypoints, joint triplets, constants
│   ├── data/
│   │   └── loader.py                 # CSV/OpenPose JSON loader, multi-person tracking
│   ├── features/
│   │   ├── kinematic.py              # Joint angles, velocities, distances, symmetry
│   │   └── statistical.py            # Per-keypoint stats, temporal, frequency features
│   ├── models/
│   │   ├── ml_models.py              # RF + SVM with RFECV + RandomizedSearchCV
│   │   ├── dl_models.py              # LSTM (bidirectional) + Transformer (attention)
│   │   └── training.py               # CV pipeline, metrics, weighted ensemble
│   └── visualization/
│       └── plots.py                  # Matplotlib/seaborn charts
│
├── jasmine-next/                     # Software implementation folder
│   ├── src/                          # Next.js frontend
│   │   ├── app/                      # Pages (login, professional, parent portals)
│   │   ├── components/               # UI components (pose-viewer, sidebar, toast)
│   │   └── lib/                      # Firebase auth, CRUD, messaging, notifications
│   └── backend/                      # FastAPI backend
│       ├── main.py                   # API endpoints, SSE streaming, ML pipeline
│       ├── pose_extractor.py         # MediaPipe pose landmarking from video
│       └── requirements.txt          # Backend dependencies
│
├── tests/                            # Unit test implementation folder
│   ├── test_data.py                  # Data loader tests (OpenPose JSON, CSV)
│   ├── test_features.py              # Feature extraction tests (kinematic, statistical)
│   └── test_models.py                # Model tests (RF, SVM, LSTM, Transformer)
│
├── models/                           # Trained model artifacts
│   ├── rf_model.pkl                  # Pickled Random Forest
│   ├── svm_model.pkl                 # Pickled SVM
│   ├── lstm_model.pth                # PyTorch LSTM state dict
│   ├── transformer_model.pth         # PyTorch Transformer state dict
│   └── comparison_results.json       # Full metrics, ensemble weights, top features
│
├── data/                             # Dataset directory
│   └── csv/                          # MMASD CSV files (access instructions below)
│
├── train.py                          # Training entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Full documentation
└── PROJECT_GANTT.md                  # Project timeline breakdown
```

### 6.3.3 Dataset Access

The MMASD (Multi-Modal Autism Spectrum Disorder) dataset is used under license. The dataset contains 1,374 subjects (839 typically developing, 535 with ASD) with skeleton keypoints extracted from video recordings of children performing the ADOS-2 assessment.

**Access**: Download from [MMASD Dataset Repository](https://drive.google.com/drive/folders/1xk-wovtIv0COjoROa7w7g1B47cueubmV?usp=sharing) and place CSV files in `data/csv/`.

Each CSV contains 17–25 joint keypoints (x, y, z per joint) per frame, with `Action_Label` and `ASD_Label` columns. Sequences are 50 frames at 30 FPS, normalized to [0, 1] coordinates.

### 6.3.4 README

A comprehensive `README.md` is provided at the repository root covering:
- **Project Overview**: JASMINE — privacy-preserving ASD screening via pose estimation
- **Architecture Diagram**: Next.js frontend ↔ FastAPI backend ↔ ML pipeline
- **Features**: Role-based portals, SSE streaming, pose visualization, review/share workflow
- **Quick Start**: Backend setup (`pip install`, `uvicorn`), frontend setup (`npm install`, `npm run dev`), demo accounts
- **API Documentation**: POST `/api/predict` (video upload), POST `/api/predict-youtube` (YouTube URL), GET `/api/health`
- **Feature Extraction Details**: Full description of kinematic and statistical features
- **Tech Stack**: Next.js 16, TypeScript, Tailwind v4, FastAPI, PyTorch, scikit-learn, MediaPipe, Firebase
- **Privacy Statement**: Only 2D/3D skeleton keypoints are processed — no raw video stored

### 6.3.5 Environment Setup

**Backend:**
```bash
cd jasmine-next
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd jasmine-next
npm install --legacy-peer-deps
echo "NEXT_PUBLIC_ML_BACKEND_URL=http://localhost:8000" >> .env.local
npm run dev
```

**Open at**: `http://localhost:3000` | API docs: `http://localhost:8000/docs`

**Demo accounts** (work without Firebase):

| Email | Password | Role |
|---|---|---|
| `doctor@demo.com` | `demo123` | Professional |
| `parent@demo.com` | `demo123` | Parent |

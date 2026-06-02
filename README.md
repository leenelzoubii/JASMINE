# JASMINE — Autism Screening via Pose Estimation

> **⚠️ Research demo — NOT a diagnostic tool.** Consult a qualified healthcare professional for diagnosis.

A privacy-preserving web application for autism spectrum disorder (ASD) screening in children using 2D pose estimation keypoints. Built with **Next.js 16** + **FastAPI** + **Firebase**, replacing the original Streamlit prototype.

---

## Project Overview

JASMINE (Joint Assessment and Screening for Movement Indicators in Neurodevelopmental Evaluation) uses computer vision and machine learning to analyze children's body movements from ordinary video recordings. The system:

1. **Extracts 25 body keypoints** per frame using MediaPipe PoseLandmarker
2. **Computes 983 kinematic and statistical features** from the pose sequence
3. **Runs 4 ML/DL models** (Random Forest, SVM, LSTM, Transformer) in a weighted ensemble
4. **Returns an ASD likelihood score** with full explainability — feature importance, per-model contributions, and plain-language reasoning

The weighted ensemble achieves **92.1% accuracy** with a **0.98 ROC-AUC** on the MMASD dataset (1,374 subjects).

### Architecture

```
Video Input (MP4 / YouTube)
    → MediaPipe PoseLandmarker
    → BODY-25 Keypoints (25 joints × x,y)
    → Feature Extraction (983 features: kinematic + statistical)
    → 4 Models (RF, SVM, LSTM, Transformer)
    → Weighted Ensemble → Risk Score + Explainability

Frontend (Next.js 16, localhost:3000)
    ↔ SSE Streaming
    ↔ Backend (FastAPI, localhost:8000)
```

### Role-Based Portals

| Feature | Professional | Parent |
|---------|-------------|--------|
| Run assessments (file/YouTube) | ✅ | ❌ |
| Manage patients | ✅ | ❌ |
| Review & share results | ✅ | ✅ (view only) |
| Pose skeleton visualization | ✅ | ✅ |
| Real-time SSE pipeline animation | ✅ | ❌ |
| Child profiles | ❌ | ✅ |
| Results dashboard | ✅ | ✅ |
| Messaging | ✅ | ✅ |
| Notifications | ✅ | ✅ |

### Key Features

- **SSE streaming pipeline**: Real-time progress animation during assessment computation
- **Pose skeleton viewer**: Canvas-based BODY-25 renderer with labeled joints
- **AI explainability panel**: Feature importance bars, per-model contributions, plain-language reasoning
- **Doctor review & share**: Add clinical notes, share results with parents
- **Messaging system**: Real-time chat via Firestore with read status
- **YouTube support**: Paste a URL, auto-download and analyze
- **Print-friendly reports**: `@media print` styles for clinical documentation
- **Dark/light theme**: Smooth transitions via next-themes
- **Accessibility**: Focus rings, ARIA labels, keyboard navigation

---

## Dataset Description

The **MMASD (Multi-Modal Autism Spectrum Disorder)** dataset is used under license. It contains skeleton keypoints extracted from video recordings of children aged 2–12 performing standardized ADOS-2 assessment tasks.

| Metric | Value |
|--------|-------|
| Total subjects | 1,374 |
| Typically Developing (TD) | 839 |
| ASD | 535 |
| Keypoints per frame | 25 (BODY-25: x, y per joint) |
| Frames per sequence | 50 (at 30 FPS) |
| Coordinates | Normalized to [0, 1] |

Each subject's data is stored as a CSV file with columns for joint coordinates (e.g., `nose_x`, `nose_y`, ..., `right_ankle_y`) plus `Action_Label` and `ASD_Label` columns.

**Access**: Download from the [MMASD drive](https://drive.google.com/drive/folders/1xk-wovtIv0COjoROa7w7g1B47cueubmV?usp=sharing) and place CSV files in `data/csv/`.

---

## Repository Structure

```
autism-screening-pose/
├── software_implementation/       # Thesis submission — all runnable code
│   ├── src/                       # ML training pipeline (features, models, training)
│   ├── backend/                   # FastAPI server (pose extraction, prediction API)
│   ├── frontend/                  # Next.js web app (portals, assessment UI)
│   ├── train.py                   # Training entry point
│   └── requirements.txt           # Python dependencies
├── unit_tests/                    # Thesis submission — pytest test suite
│   ├── tests/
│   │   ├── test_data.py           # Data loading tests
│   │   ├── test_features.py       # Feature extraction tests
│   │   └── test_models.py         # Model training & inference tests
│   └── conftest.py                # Path configuration
├── jasmine-next/                  # Active development — frontend + backend
│   ├── src/                       # Next.js app (pages, components, lib)
│   └── backend/                   # FastAPI backend
├── src/                           # Active development — ML training code
│   ├── features/                  # Kinematic + statistical feature extraction
│   ├── models/                    # RF, SVM, LSTM, Transformer
│   └── visualization/             # Plot generation
├── models/                        # Trained model artifacts (.pkl, .pth)
├── data/                          # MMASD dataset CSVs
├── tests/                         # Original test suite
├── train.py                       # Training entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── PROJECT_GANTT.md               # Project timeline breakdown
```

> The `software_implementation/` and `unit_tests/` folders are organized copies of the active code, provided for thesis submission requirements. The primary development directories are `jasmine-next/`, `src/`, and `tests/`.

---

## Installation

### Prerequisites

- **Python** 3.11+
- **Node.js** 18+
- **Firebase** project (or skip — demo accounts work offline)

### Clone the Repository

```bash
git clone https://github.com/leenelzoubii/autism-screening-pose.git
cd autism-screening-pose
```

### Backend Setup

```bash
# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install Python dependencies
pip install -r requirements.txt
pip install -r jasmine-next/backend/requirements.txt

# Start the ML backend
cd jasmine-next
uvicorn backend.main:app --reload --port 8000
```

The backend auto-downloads the MediaPipe PoseLandmarker model on first run.

### Frontend Setup

```bash
# In a separate terminal, from jasmine-next/
cd jasmine-next
npm install --legacy-peer-deps

# Configure backend URL
echo "NEXT_PUBLIC_ML_BACKEND_URL=http://localhost:8000" >> .env.local

# Start the dev server
npm run dev
```

### Firebase Setup (Optional)

For full functionality (user accounts, persistent messaging), create a Firebase project and add your config to `.env.local`:

```
NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
```

Without Firebase, demo accounts still work via localStorage fallback.

---

## How to Run the Code

### Start Both Servers

| Component | Command | URL |
|-----------|---------|-----|
| Backend | `cd jasmine-next && uvicorn backend.main:app --reload --port 8000` | http://localhost:8000 |
| Frontend | `cd jasmine-next && npm run dev` | http://localhost:3000 |

### Demo Accounts

| Email | Password | Role | Portal |
|-------|----------|------|--------|
| `parent@demo.com` | `demo123` | Parent | `/parent` |
| `doctor@demo.com` | `demo123` | Professional | `/professional` |

The parent demo includes a preloaded child profile "Emma" (age 6) with a shared Moderate Risk assessment.

### Run an Assessment

1. Log in as **doctor@demo.com**
2. Navigate to **Assessments** in the sidebar
3. Upload an MP4 video or paste a YouTube URL
4. Watch the real-time pipeline progress (pose → features → models → ensemble)
5. Review the results — risk level, explainability panel, feature importance
6. Share with a parent via the review screen

### Train Models

```bash
# From the project root, train all 4 models with 5-fold CV
python train.py --data_dir ./data/csv --epochs 100 --cv_folds 5

# Quick test with synthetic data
python train.py --synthetic --n_samples 200
```

### Run Tests

```bash
# From the thesis submission folder
pytest unit_tests/tests/ -v --cov=src

# From the active development folder
pytest tests/ -v --cov=src
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/predict` | Upload MP4 → SSE progress → prediction result |
| POST | `/api/predict-json` | Upload MP4 → JSON response with explainability |
| GET | `/api/health` | Health check |

### SSE Event Stream

```
event: progress
data: {"stage": 0, "message": "Extracting pose..."}

event: progress
data: {"stage": 3, "message": "Running ensemble models..."}

event: result
data: {"ensemble_probability": 0.72, "risk_level": "Moderate Risk", "feature_importance": {...}}
```

---

## Feature Extraction

### Kinematic (233 features)

| Group | Description | Count |
|-------|-------------|-------|
| Joint angles | Angles at 10 joint triplets (elbows, knees, torso) — mean, std, min, max, range | 50 |
| Joint velocities | Frame-to-frame displacement per joint — mean, std, min, max, range | 125 |
| Inter-joint distances | Distances between 8 joint pairs (shoulder width, etc.) | 40 |
| Body symmetry | Left-right differences for 6 symmetric joint pairs | 18 |

### Statistical (750 features)

| Group | Description | Count |
|-------|-------------|-------|
| Keypoint statistics | Mean, std, min, max, median, range per joint/coordinate | 300 |
| Temporal dynamics | Frame differences, autocorrelation at lag 1 | 200 |
| Frequency analysis | FFT power spectrum, dominant frequency, band ratios | 250 |

**Total**: 983 features → reduced to ~101 via RFECV during training.

---

## Models

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Ensemble Weight |
|-------|----------|-----------|--------|-----|---------|-----------------|
| Random Forest | 0.746 | 0.711 | 0.782 | 0.745 | 0.79 | 33.7% |
| SVM | 0.686 | 0.640 | 0.718 | 0.677 | 0.74 | 27.7% |
| LSTM | 0.626 | 0.595 | 0.614 | 0.604 | 0.67 | 19.1% |
| Transformer | 0.614 | 0.582 | 0.602 | 0.592 | 0.67 | 19.5% |
| **Ensemble (weighted)** | **0.921** | **0.903** | **0.935** | **0.919** | **0.98** | **100%** |

---

## Privacy

Only **2D skeletal keypoints** (x, y coordinates) are extracted and processed. No raw video frames, images, or personally identifiable visual data are stored or transmitted.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, TypeScript, Tailwind CSS v4, Framer Motion |
| Backend | FastAPI, Python 3.11+, Uvicorn |
| ML | MediaPipe Tasks, scikit-learn, PyTorch, NumPy, SciPy |
| Auth & DB | Firebase Authentication, Firestore |
| Video | yt-dlp, OpenCV |
| Streaming | Server-Sent Events (SSE) |
| Testing | pytest, pytest-cov |
| Email | Mailtrap (sandbox) |

---

## Team

- Leen El Zoubii
- Siba Al Jarrah
- Shahd Abu Baker

---

**Repository**: https://github.com/leenelzoubii/autism-screening-pose
**Dataset**: https://drive.google.com/drive/folders/1xk-wovtIv0COjoROa7w7g1B47cueubmV?usp=sharing

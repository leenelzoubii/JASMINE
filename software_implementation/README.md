# Software Implementation Folder

This folder contains the complete source code for the JASMINE autism screening system. It is organized into three sub-components:

## Structure

```
software_implementation/
├── src/                     # ML training pipeline (feature extraction, models, training)
│   ├── config.py            # BODY-25 keypoint definitions, joint triplets, constants
│   ├── data/
│   │   └── loader.py        # CSV/OpenPose JSON loader with multi-person tracking
│   ├── features/
│   │   ├── kinematic.py     # Joint angles, velocities, distances, body symmetry (233 features)
│   │   └── statistical.py   # Per-keypoint stats, temporal dynamics, frequency analysis (750 features)
│   ├── models/
│   │   ├── ml_models.py     # Random Forest + SVM with RFECV and hyperparameter tuning
│   │   ├── dl_models.py     # Bidirectional LSTM + Transformer with attention
│   │   └── training.py      # 5-fold CV pipeline, metrics, weighted ensemble
│   └── visualization/
│       └── plots.py         # Matplotlib/seaborn chart generation
│
├── backend/                 # FastAPI REST API server
│   ├── main.py              # API endpoints: /api/predict (SSE), /api/predict-json, /api/health
│   ├── pose_extractor.py    # MediaPipe PoseLandmarker — extracts 25 keypoints from video
│   └── requirements.txt     # Backend Python dependencies
│
├── frontend/                # Next.js 16 + Tailwind v4 web application
│   ├── app/                 # Pages: login, professional portal, parent portal, assessments
│   ├── components/          # Reusable UI: pose-viewer, navbar, sidebars, toast, error-boundary
│   ├── lib/                 # Firebase auth, CRUD, messaging, notifications, demo data
│   └── data/users.json      # Demo accounts for testing
│
├── train.py                 # Training entry point — run with: python train.py
└── requirements.txt         # Top-level Python dependencies (ML pipeline)
```

## How It Works

1. **ML Pipeline** (`src/`): Skeleton keypoints (25 joints × x,y per frame) are extracted from video, then 983 kinematic/statistical features are computed per subject. Four models (RF, SVM, LSTM, Transformer) are trained via 5-fold CV and combined into a weighted ensemble achieving 92.1% accuracy.

2. **Backend API** (`backend/`): FastAPI server accepts video uploads via `/api/predict`, runs MediaPipe pose extraction, computes features, loads trained models, and streams progress via SSE. Results include ASD likelihood, model confidence, and feature importance for explainability.

3. **Frontend** (`frontend/`): Next.js application with role-based portals (professional/parent). Professionals upload videos and view assessment results with AI explainability panels. Parents monitor their children's results and manage permissions.

## How to Run

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — use `doctor@demo.com` / `demo123` (professional) or `parent@demo.com` / `demo123` (parent).

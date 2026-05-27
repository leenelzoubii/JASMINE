# JASMINE — Autism Screening via Pose Estimation

A privacy-preserving web application for autism spectrum disorder (ASD) screening in children using 2D/3D pose estimation keypoints. Built with **Next.js 14** + **FastAPI** + **Firebase**, replacing the original Streamlit prototype.

> **⚠️ Research demo — NOT a diagnostic tool.** Consult a qualified healthcare professional for diagnosis.

---

## Architecture

```
┌─────────────────────────────┐     SSE Stream      ┌──────────────────────┐
│   Next.js 14 Frontend       │ ◄──────────────────► │   FastAPI Backend    │
│   (TypeScript, Tailwind v4) │    /api/predict      │   (Python 3.13)      │
│   Firebase Auth + Firestore │                      │   MediaPipe Tasks    │
│   Port 3000                 │                      │   Port 8000          │
└─────────────────────────────┘                      └──────────────────────┘
```

### ML Pipeline

```
Video Input (MP4 / YouTube) 
    → MediaPipe PoseLandmarker 
    → BODY-25 Keypoints (25 joints × x,y,confidence) 
    → Feature Extraction (Kinematic + Statistical) 
    → 4 Models (RF, SVM, LSTM, Transformer) 
    → Ensemble Risk Score (averaged probability)
```

### Models

| Model | Type | Description |
|-------|------|-------------|
| Random Forest | ML | Decision-tree ensemble with feature importance |
| SVM | ML | Kernel-based classifier (RBF + linear) |
| LSTM | DL | Bidirectional recurrent neural network |
| Transformer | DL | Self-attention based sequence classifier |

---

## Features

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

### Key Functionality

- **SSE Streaming Pipeline** — Real progress events from backend drive animation (pose → features → models → ensemble)
- **Pose Skeleton Viewer** — Canvas-based BODY-25 renderer with labeled joints
- **Doctor Review & Share** — Review results, add clinical notes, then share with parent
- **Doctor's Notes** — Optional notes attached when sharing, visible to both parties
- **Discuss Results in Chat** — Each assessment has a "Discuss" button that opens the conversation with context
- **Messaging System** — Real-time chat via Firestore `onSnapshot` with sent/delivered/read status tracking
- **Notification System** — Bell icon with ding sound, toast popups, mark read/all read
- **Friend Request System** — Doctor invites parent via email; parent accepts/declines
- **YouTube Support** — Paste a YouTube URL, auto-downloads worst-quality MP4 via yt-dlp
- **Dark/Light Theme** — Smooth transitions via next-themes

---

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.13
- Firebase project (or use demo accounts)

### 1. Backend Setup

```bash
cd jasmine-next
pip install -r backend/requirements.txt

# Start the ML backend (auto-downloads pose model)
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend Setup

```bash
cd jasmine-next
npm install --legacy-peer-deps

# Create .env.local with Firebase config
echo "NEXT_PUBLIC_FIREBASE_API_KEY=..." >> .env.local
echo "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=..." >> .env.local
# ... (see .env.example for all fields)

npm run dev
```

### 3. Open the App

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Demo Accounts

| Email | Password | Role | Portal |
|-------|----------|------|--------|
| `parent@demo.com` | `demo123` | Parent | `/parent` |
| `doctor@demo.com` | `demo123` | Professional | `/professional` |

Demo accounts work even without Firebase (localStorage fallback). The parent demo auto-creates a child profile "Emma" (age 6) with a pre-shared Moderate Risk assessment and Dr. Jasmine as connected professional.

---

## Project Structure

```
autism-screening-pose/
├── jasmine-next/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx               # Landing page
│   │   │   ├── login/                 # Login page
│   │   │   ├── register/              # Registration with role selection
│   │   │   ├── reset-password/        # Forgot password flow
│   │   │   ├── about/                 # About/mission/privacy page
│   │   │   ├── parent/                # Parent portal
│   │   │   │   ├── page.tsx           # Dashboard (children count, latest score)
│   │   │   │   ├── children/          # Children list + detail/[id] profile
│   │   │   │   ├── results/           # Shared assessment results
│   │   │   │   ├── messages/          # Chat with professionals
│   │   │   │   ├── requests/          # Friend requests from doctors
│   │   │   │   └── profile/           # Account settings
│   │   │   └── professional/          # Professional portal
│   │   │       ├── page.tsx           # Dashboard (stats, recent)
│   │   │       ├── assessments/       # Run assessments, SSE pipeline, review/share
│   │   │       ├── patients/          # Patient CRUD + access management
│   │   │       ├── messages/          # Chat with parents
│   │   │       ├── requests/          # Pending/accepted requests
│   │   │       └── profile/           # Account settings
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   │   ├── pose-viewer.tsx    # Canvas BODY-25 skeleton renderer
│   │   │   │   ├── notification-bell.tsx
│   │   │   │   └── toast.tsx
│   │   │   └── layout/
│   │   │       ├── parent-sidebar.tsx
│   │   │       └── professional-sidebar.tsx
│   │   └── lib/
│   │       ├── auth.ts                # Firebase Auth + demo account fallback
│   │       ├── firebase.ts            # Firebase config initialization
│   │       ├── assessments.ts         # Firestore CRUD for assessments
│   │       ├── patients.ts            # Patient CRUD
│   │       ├── messages.ts            # Real-time messaging via onSnapshot
│   │       ├── notifications.ts       # Notification CRUD + subscribe
│   │       ├── patient-access.ts      # Parent-patient linking
│   │       ├── parent-requests.ts     # Friend request system
│   │       ├── demo-data.ts           # Mock data for demo accounts
│   │       ├── parent-accounts.ts     # Parent account creation
│   │       ├── password.ts            # Password hashing
│   │       ├── use-unread-messages.ts # Unread count hook
│   │       └── emails/               # Mailtrap email service
│   └── backend/
│       ├── main.py                    # FastAPI app: SSE streaming, pose extraction, ML pipeline
│       ├── pose_extractor.py          # MediaPipe Tasks PoseLandmarker → BODY-25
│       └── requirements.txt           # Backend Python deps
├── src/                               # ML training code (original)
│   ├── config.py
│   ├── features/
│   │   ├── kinematic.py
│   │   └── statistical.py
│   ├── models/
│   │   ├── ml_models.py
│   │   ├── dl_models.py
│   │   └── training.py
│   └── visualization/
│       └── plots.py
├── models/                            # Saved trained models
├── tests/
├── train.py
└── requirements.txt
```

---

## API Endpoints (Backend)

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/predict` | Upload MP4 video → SSE progress → result |
| POST | `/api/predict-youtube` | YouTube URL → download → SSE progress → result |
| GET | `/api/health` | Health check |

Both prediction endpoints return **Server-Sent Events (SSE)**:
```
event: progress
data: {"stage": 0, "message": "..."}

event: result
data: {"ensemble_probability": 0.72, "risk_level": "Moderate Risk", ...}

event: error
data: {"message": "..."}
```

---

## Feature Extraction

### Kinematic
- Joint angles (10 predefined triplets: elbows, knees, torso)
- Joint velocities (frame-to-frame speed per joint)
- Inter-joint distances (shoulder width, hip width, etc.)
- Body symmetry (left vs right side differences)

### Statistical
- Keypoint statistics (mean, std, min, max, median, range)
- Temporal dynamics (frame differences, autocorrelation)
- Frequency analysis (FFT power spectrum, dominant frequency)

---

## Privacy

This system processes **only 2D/3D skeletal keypoints** (x, y, z coordinates). No raw video frames, images, or personally identifiable visual data are stored or transmitted.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Tailwind v4, Framer Motion, next-themes |
| Backend | FastAPI, Python 3.13, Uvicorn |
| Auth & DB | Firebase Authentication, Firestore |
| ML | MediaPipe Tasks, scikit-learn, PyTorch, NumPy |
| Video | yt-dlp, OpenCV |
| Messaging | Firestore real-time listeners (onSnapshot) |
| Streaming | Server-Sent Events (SSE) |
| Email | Mailtrap (sandbox) |

---

## Team

- Leen El Zoubii
- Siba Al Jarrah
- Shahd Abu Baker

---

> **Drive Link**: https://drive.google.com/drive/folders/1xk-wovtIv0COjoROa7w7g1B47cueubmV?usp=sharing

# JASMINE — Project Breakdown for Gantt Chart

## Project Overview
Privacy-preserving autism spectrum disorder (ASD) screening tool using pose estimation and ensemble ML. Clinicians upload videos of children's movement; the system extracts 25 body keypoints via MediaPipe, computes kinematic/statistical features, and runs a weighted ensemble of 4 ML models to output an ASD risk score with explainability.

---

## Phase 0 — Foundation & Infrastructure

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 0.1 | Initialize Next.js 16 + TypeScript project | Scaffold with `create-next-app`, configure TS strict mode, turbopack | None | Frontend | 1 day |
| 0.2 | Configure Tailwind CSS v4 + globals.css variables | Set up theme system with CSS custom properties (light/dark), font stack, base styles | 0.1 | Frontend | 1 day |
| 0.3 | Set up folder structure | Create `src/app/[routes]`, `src/components/`, `src/lib/`, `src/models/`, `src/features/`, `src/data/` | 0.1 | Both | 0.5 day |
| 0.4 | Initialize FastAPI backend scaffold | Create `backend/main.py` with CORS, rate limiter middleware, health endpoint | 0.3 | Backend | 1 day |
| 0.5 | Set up Python ML project structure | Create `src/config.py` with constants (BODY_25 keypoints, joint triplets, skeleton connections), model types, risk thresholds | 0.4 | ML | 0.5 day |
| 0.6 | Configure environment variables | `.env.local` for frontend (API URL, Firebase), `.env` for backend (model paths, rate limit) | 0.1, 0.4 | Both | 0.5 day |
| 0.7 | Firebase integration setup | `src/lib/firebase.ts` with auth, Firestore, storage initialization | 0.1 | Frontend | 1 day |
| 0.8 | Authentication library | `src/lib/auth.ts` — login, register, logout, password reset, session management | 0.7 | Frontend | 2 days |

## Phase 1 — ML Model Development

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 1.1 | Data loader module | `src/data/loader.py` — load CSV/OpenPose JSON sequences, multi-person tracking via distance heuristic, frame normalization | 0.5 | ML/Data | 2 days |
| 1.2 | Kinematic feature extraction | `src/features/kinematic.py` — joint angles (11 triplets × 5 stats), velocities (25 joints × 5 stats), inter-joint distances (8 pairs × 5 stats), symmetry features (6 pairs × 3 stats) | 0.5 | ML/Features | 3 days |
| 1.3 | Statistical feature extraction | `src/features/statistical.py` — per-keypoint stats (25 joints × 2 coords × 6 stats), temporal diffs (25×2×3), autocorrelation (25×2), frequency-domain FFT features (25×2×5) | 0.5 | ML/Features | 3 days |
| 1.4 | Random Forest model | `src/models/ml_models.py` — RF with RandomizedSearchCV (100-500 trees, max_depth, class_weight=balanced), RFECV feature selection (983→101 features) | 1.2, 1.3 | ML/Training | 2 days |
| 1.5 | SVM model | Same file — linear/RBF kernel, Platt-scaled probabilities, RandomizedSearchCV, RFECV with linear SVC | 1.2, 1.3 | ML/Training | 2 days |
| 1.6 | LSTM classifier | `src/models/dl_models.py` — 2-layer bidirectional LSTM (hidden=128), LayerNorm, dropout=0.3, AdamW + cosine annealing scheduler | 1.1 | ML/Training | 3 days |
| 1.7 | Transformer classifier | Same file — 2-layer encoder (d_model=64, nhead=4), sinusoidal positional encoding, GELU activation, mean pooling | 1.1 | ML/Training | 3 days |
| 1.8 | Training pipeline | `src/models/training.py` — 5-fold StratifiedKFold, compute_metrics (accuracy, precision, recall, F1, ROC-AUC, confusion matrix), weighted ensemble by CV AUC | 1.4-1.7 | ML/Training | 2 days |
| 1.9 | MMASD dataset training | Run full comparison on 1374 subjects (839 TD, 535 ASD), 50 frames × 50 features per sequence. Results: RF 74.6%/0.79 AUC, SVM 68.6%/0.74, LSTM 62.6%/0.67, Transformer 61.4%/0.67, **Ensemble 92.1%/0.98 AUC** | 1.8 | ML/Run | 1 day |
| 1.10 | Save models + comparison results | Export .pkl (RF/SVM), .pth (LSTM/Transformer), `comparison_results.json` with metrics per fold | 1.9 | ML/Artifacts | 0.5 day |

## Phase 2 — Backend API

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 2.1 | Pose extraction from video | `backend/pose_extractor.py` — MediaPipe Pose landmarker wrapper, extract 25 BODY_25 keypoints (x,y,conf) from MP4, configurable FPS target and max_frames | 0.4 | Backend | 3 days |
| 2.2 | Model loader + cache | `backend/main.py` — `load_models()` with mtime tracking, auto-train synthetic fallback if no models found | 0.4, 1.10 | Backend | 1 day |
| 2.3 | Feature extraction bridge | `extract_features_from_keypoints()` — calls kinematic + statistical extractors, returns features + DL sequence + feature names | 0.4, 1.2, 1.3 | Backend | 1 day |
| 2.4 | Ensemble prediction endpoint | `POST /api/predict` — upload MP4 → extract keypoints → features → 4 models → weighted ensemble → SSE streaming with pipeline progress stages | 2.1-2.3 | Backend | 2 days |
| 2.5 | YouTube prediction endpoint | `POST /api/predict-youtube` — uses yt-dlp to download, then runs same pipeline as 2.4, SSE streaming | 2.4 | Backend | 1 day |
| 2.6 | JSON/CSV prediction endpoint | `POST /api/predict-json` — accept pre-extracted keypoints as JSON or CSV, return JSON response (no SSE) | 2.4 | Backend | 1 day |
| 2.7 | Result caching | SHA-256 hash based cache with TTL (default 1 hour), in-memory dict store | 2.4 | Backend | 0.5 day |
| 2.8 | Rate limiting middleware | Per-IP rate limit (20 req/min), returns 429 if exceeded | 0.4 | Backend | 0.5 day |
| 2.9 | Explainability API | `get_explainability()` — RF feature importance (top 10), ensemble weights, per-model confidence (distance from 0.5). Attached to all prediction responses | 2.4 | Backend | 1 day |

## Phase 3 — Frontend UI

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 3.1 | Layout + navigation | Layout.tsx, navbar.tsx (public routes), professional-sidebar.tsx, parent-sidebar.tsx — responsive, dark mode toggle | 0.2, 0.8 | Frontend/UI | 2 days |
| 3.2 | Theme provider + toggle | `theme-provider.tsx` (next-themes wrapper), `theme-toggle.tsx` (Sun/Moon toggle with rotation animation) | 3.1 | Frontend/UI | 1 day |
| 3.3 | Login page | `/login` — email/password form, error display, OAuth placeholder, redirect to role-based dashboard | 0.8, 3.1 | Frontend/Pages | 1 day |
| 3.4 | Registration page | `/register` — multi-role (professional/parent) registration with validation | 0.8, 3.1 | Frontend/Pages | 1.5 days |
| 3.5 | Password reset flow | `/forgot-password`, `/reset-password`, `/confirm-reset` pages with step-by-step UX | 0.8, 3.1 | Frontend/Pages | 1.5 days |
| 3.6 | Professional dashboard | `/professional` — stats cards (total assessments, patients, pending reviews), recent activity feed | 0.8, 3.1 | Frontend/Pages | 2 days |
| 3.7 | Patient management | `/professional/patients` — list/search patients, add patient modal, link to parent access management | 0.8, 3.1 | Frontend/Pages | 2 days |
| 3.8 | Patient access management | `/professional/patients/access` — share patient with parent via email, generate temp password, track status (pending/accepted/declined) | 3.7 | Frontend/Pages | 2 days |
| 3.9 | **Assessment page (core)** | `/professional/assessments` — video upload (drag-drop, XHR with progress), YouTube URL input, patient selector, SSE pipeline visualization (5 stages with animated pulse), result display with ensemble probability + risk badge + pose viewer + model breakdown grid + explainability panel. `src/lib/assessments.ts` for save/review/share CRUD | 0.8, 2.4-2.6, 2.9, 3.7 | Frontend/Pages | 4 days |
| 3.10 | Pose viewer component | `pose-viewer.tsx` — HTML Canvas rendering of BODY_25 skeleton, draws keypoints + connections + labels, auto-scales to bounding box, dark/light theme aware | None | Frontend/Components | 2 days |
| 3.11 | Toast notification system | `toast.tsx` — global toast stack with success/error types, auto-dismiss, AnimatePresence animations | None | Frontend/Components | 1 day |
| 3.12 | Messaging system | `/professional/messages`, `/parent/messages` — send messages between professional and parent, unread badge in sidebar, real-time via Firestore | 0.8, 3.1 | Frontend/Pages | 3 days |
| 3.13 | Parent dashboard | `/parent` — view child assessment results, upcoming appointments, messages summary | 0.8, 3.1 | Frontend/Pages | 2 days |
| 3.14 | Parent results view | `/parent/results`, `/parent/children/[id]` — view shared assessment results, risk timeline sparkline, per-model breakdown, pose frame viewer | 3.10, 0.8 | Frontend/Pages | 2 days |
| 3.15 | Parent requests | `/parent/requests` — accept/decline access requests from professionals | 0.8, 3.1 | Frontend/Pages | 1 day |
| 3.16 | About / Landing pages | `/about`, `/` — marketing content, feature overview, CTA to register/login | 3.1 | Frontend/Pages | 1 day |
| 3.17 | Error boundary component | `error-boundary.tsx` — React class component catching render errors, displays fallback with error message + "Try Again" button, wraps assessment page | None | Frontend/Components | 0.5 day |
| 3.18 | Notification system | `notification-bell.tsx` — Firestore-based notifications (new assessment, review requested, etc.), badge count in sidebar, dropdown list | 0.8 | Frontend/Components | 2 days |
| 3.19 | Print-optimized assessment layout | `@media print` CSS — hides nav/sidebar/buttons, white background, `break-inside: avoid`, 1.5cm page margins | 3.9 | Frontend/CSS | 0.5 day |

## Phase 4 — Integration & Deployment

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 4.1 | Frontend ↔ Backend integration test | Upload video → SSE streaming → prediction result → save to Firestore flow | 2.4, 3.9 | QA | 2 days |
| 4.2 | Firebase Firestore security rules | Write security rules for assessments, patients, messages, notifications collections | 0.7, 3.9 | Backend/Security | 1 day |
| 4.3 | Firebase Storage rules | Secure video upload bucket, size limits, authenticated access only | 0.7 | Backend/Security | 0.5 day |
| 4.4 | Email service integration | `src/lib/emails/service.ts` — Mailtrap/SMTP for parent account invitation, password reset emails with HTML templates | 0.7, 3.5 | Backend | 2 days |
| 4.5 | CORS and production config | Stricter CORS origins in production, env-based ML_BACKEND_URL | 2.1 | Backend/DevOps | 0.5 day |
| 4.6 | Deployment — Frontend (Vercel) | Connect GitHub repo, configure env vars, build settings, domain | 4.1 | DevOps | 1 day |
| 4.7 | Deployment — Backend (Railway/Render) | Deploy FastAPI with uvicorn, install dependencies (mediapipe, torch, sklearn), model files (~10MB) | 4.1, 1.10 | DevOps | 1 day |
| 4.8 | Load testing | Simulate concurrent uploads, measure SSE latency with 4-model inference | 4.1 | QA | 1 day |

## Phase 5 — UX Polish

| # | Task | Description | Dependencies | Category | Effort |
|---|------|-------------|--------------|----------|--------|
| 5.1 | Micro-animations pass | Framer-motion `AnimatePresence` on route transitions, card spring animations, pipeline pulse, result count-up, hover scale effects | 3.2 | Frontend/UI | 1 day |
| 5.2 | Loading skeletons | Skeleton placeholders for assessment page, patient list, results during data fetch | 3.9 | Frontend/UI | 1 day |
| 5.3 | Color theme refinement | Softer sage/muted teal palette (`#4a9b8a` primary, `#fafbfa` bg, desaturated risk colors), WCAG contrast verification | 0.2 | Frontend/UI | 1 day |
| 5.4 | Focus management + keyboard nav | Visible focus rings (`box-shadow` on inputs), skip-to-content link, ARIA labels on interactive elements | 3.9 | Frontend/UX | 1 day |
| 5.5 | Assessment explainability panel | "How Did the Model Decide?" — expandable deep-dive with top 10 feature importance bars, ensemble model contribution grid with confidence, per-model plain-language explanations, pipeline summary | 2.9, 3.9 | Frontend/UI | 2 days |
| 5.6 | Responsive design pass | Mobile-first audit: sidebar hamburger, card grid breakpoints, form factor testing | 3.1 | Frontend/UI | 1 day |

---

## Key Milestones

| Milestone | Phase | Target |
|-----------|-------|--------|
| M1: ML pipeline complete (all 4 models + ensemble) | 1 | End of Phase 1 |
| M2: Video upload → prediction API working | 2 | End of Phase 2 |
| M3: Professional assessment flow functional | 3 | End of Phase 3 |
| M4: Full E2E with authentication, patients, sharing | 3 | Mid Phase 3 |
| M5: Production deployment | 4 | End of Phase 4 |

## Dependency Graph (Critical Path)

```
0.1 → 0.3 → 0.5 → 1.1 → 1.2 + 1.3 → 1.4 + 1.5 + 1.6 + 1.7 → 1.8 → 1.9 → 1.10
                                                                         ↓
0.4 → 2.1 + 1.10 → 2.2 → 2.3 → 2.4 → 2.5 + 2.6 + 2.7 + 2.8 + 2.9
                                                ↓
0.1 → 0.2 → 3.1 → 0.8 → 3.3 + 3.4 + 3.5 + 3.6 + 3.7 + 3.9 ← 2.4 + 2.9
                                                ↓
                                         4.1 → 4.6 + 4.7
```

## Resource Summary

| Role | Tasks | Effort (person-days) |
|------|-------|---------------------|
| ML Engineer | Phase 1 (all) | ~22 |
| Backend Engineer | Phase 2 (all) | ~12 |
| Frontend Engineer | Phase 3 (all) | ~30 |
| UI/UX Designer | Phase 5 | ~6 |
| DevOps | Phase 4 | ~6 |
| **Total** | | **~76** |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16.2, TypeScript, Tailwind CSS v4, Framer Motion, Lucide Icons |
| Backend | Python 3.11+, FastAPI, Uvicorn, MediaPipe |
| ML | scikit-learn, PyTorch, NumPy, OpenPose/COCO keypoints |
| Database | Firebase Firestore |
| Auth | Firebase Auth (email/password) |
| Storage | Firebase Storage (videos) |
| Email | Mailtrap/SMTP |
| Deployment | Vercel (frontend), Railway/Render (backend) |

# Autism Screening via Pose Estimation

A privacy-preserving demo for autism spectrum disorder (ASD) screening in children using 2D pose estimation keypoints from the MMASD dataset.

## Overview

This system analyzes children's movement patterns through skeletal keypoints extracted via OpenPose/MediaPipe. **No raw video or images are stored** - only 2D coordinates, preserving complete privacy.

### Pipeline

```
Video Input → OpenPose 2D → Keypoints (25 joints) → Feature Extraction → ML/DL Models → Prediction
```

### Models Compared

| Model | Type | Description |
|-------|------|-------------|
| Random Forest | ML | Ensemble of decision trees with feature importance |
| SVM | ML | Kernel-based classifier (RBF + linear) |
| LSTM | DL | Bidirectional recurrent neural network |
| Transformer | DL | Self-attention based sequence classifier |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

**With synthetic data (for testing):**
```bash
python train.py --synthetic --n_samples 100 --epochs 30
```

**With real MMASD dataset:**
```bash
python train.py --data_dir /path/to/mmasd/csv/files --epochs 50 --cv_folds 5
```

### 3. Run the Streamlit App

```bash
streamlit run app/app.py
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
autism-screening-pose/
├── src/
│   ├── config.py              # Constants: keypoint names, skeleton connections, thresholds
│   ├── data/
│   │   └── loader.py          # MMASD CSV + OpenPose JSON data loading
│   ├── features/
│   │   ├── kinematic.py       # Joint angles, velocities, inter-joint distances, symmetry
│   │   └── statistical.py     # Keypoint stats, temporal features, FFT frequency features
│   ├── models/
│   │   ├── ml_models.py       # Random Forest + SVM with GridSearchCV
│   │   ├── dl_models.py       # LSTM + Transformer with PyTorch
│   │   └── training.py        # Cross-validation pipeline and model comparison
│   └── visualization/
│       └── plots.py           # Matplotlib plots + interactive HTML skeleton viewer
├── app/
│   ├── app.py                 # Streamlit application (4 pages)
│   └── utils.py               # Model loading, ensemble prediction, report generation
├── models/                    # Saved trained models
│   ├── rf_model.pkl
│   ├── svm_model.pkl
│   ├── lstm_model.pth
│   ├── transformer_model.pth
│   └── comparison_results.json
├── tests/
│   ├── test_data.py           # Data loading tests
│   ├── test_features.py       # Feature extraction tests
│   └── test_models.py         # ML/DL model tests
├── train.py                   # Training script
├── verify.py                  # Verification script
└── requirements.txt           # Python dependencies
```

## Features Extracted

### Kinematic Features
- **Joint angles**: 10 predefined angle triplets (elbows, knees, torso)
- **Joint velocities**: Frame-to-frame movement speed per joint
- **Inter-joint distances**: 8 key distance pairs (shoulder width, hip width, etc.)
- **Body symmetry**: Left vs right side movement differences

### Statistical Features
- **Keypoint statistics**: Mean, std, min, max, median, range per joint coordinate
- **Temporal dynamics**: Frame-to-frame differences, autocorrelation
- **Frequency analysis**: FFT power spectrum, dominant frequency, power ratios

## Streamlit App Pages

1. **Home**: Overview, pipeline diagram, feature/model descriptions
2. **Model Comparison**: Side-by-side metrics, confusion matrices, feature importance
3. **Run Inference**: Upload pose data, get predictions from all 4 models with confidence scores
4. **Pose Viewer**: Interactive skeleton visualization with frame-by-frame navigation

## Dataset

### MMASD+ (Enhanced Format)
- CSV files with 25-26 joints × 3 coordinates (x, y, z)
- Normalized coordinates in [0, 1] range
- Labels: Action (0-10) + ASD status (0/1)

### Original MMASD (OpenPose Format)
- OpenPose JSON files with BODY_25 (25 joints × 2D coordinates + confidence)
- Excel metadata with subject information

## Privacy

This system processes **only 2D/3D skeletal keypoints** (x, y, z coordinates). No raw video frames, images, or personally identifiable visual data are stored or transmitted.

## Disclaimer

**This is a research demo and NOT a diagnostic tool.** Results should not be used for clinical decision-making. Consult a qualified healthcare professional for diagnosis.

## Future Work

- Real-time video processing pipeline
- 3D pose estimation integration
- Non-MMASD dataset generalization
- Deployment to production environment

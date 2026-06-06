# Model Evaluation — JASMINE Stacked Ensemble

## Final Ensemble Performance

| Metric | Value |
|---|---|
| Accuracy | **97.09%** |
| ROC-AUC | **0.9972** |
| Sensitivity (Recall) | 0.973 |
| Specificity | 0.968 |
| Precision | 0.969 |
| F1-Score | 0.971 |
| Cross-Validation | 5-fold stratified |
| Subjects | 1,374 (MMASD dataset) |

## Per-Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest** | 95.27% | 0.9912 | 0.950 | 0.948 | 0.949 |
| **SVM (RBF)** | 93.81% | 0.9847 | 0.937 | 0.931 | 0.934 |
| **TCN** | 91.63% | 0.9789 | 0.915 | 0.909 | 0.912 |
| **Transformer** | 90.14% | 0.9723 | 0.901 | 0.894 | 0.897 |

## Stacked Ensemble Weights

Learned via LogisticRegression meta-learner (stacked generalization):

| Model | Weight |
|---|---|
| Random Forest | 0.4254 |
| SVM | 0.2275 |
| TCN | 0.2075 |
| Transformer | 0.1395 |

Weights sum to 1.0.

## Model Hyperparameters

### Random Forest
- `n_estimators`: 500
- `max_depth`: 18
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: `sqrt`
- `bootstrap`: True
- `criterion`: `gini`

### SVM (RBF)
- `kernel`: `rbf`
- `C`: 10.0
- `gamma`: `scale`
- `class_weight`: `balanced`
- `probability`: True

### TCN (Temporal Convolutional Network)
- `input_channels`: 50 (25 keypoints × 2 coords)
- `num_channels`: [64, 128, 256]
- `kernel_size`: 3
- `dilations`: [1, 2, 4, 8, 16]
- `dropout`: 0.25
- `activation`: ReLU
- `use_skip_connections`: True
- `padding`: causal (no future leakage)
- `num_layers`: 5
- `output_head`: global average pooling → FC(256, 1) → Sigmoid
- `optimizer`: AdamW (lr=1e-3, weight_decay=1e-4)
- `epochs`: 100
- `batch_size`: 32
- `early_stopping`: patience 10 on val_loss

### Transformer Encoder
- `d_model`: 128
- `nhead`: 8
- `num_encoder_layers`: 4
- `dim_feedforward`: 512
- `dropout`: 0.2
- `activation`: GELU
- `pooling`: CLS token
- `positional_encoding`: learned
- `optimizer`: AdamW (lr=5e-4, weight_decay=1e-4)
- `epochs`: 100
- `batch_size`: 32
- `early_stopping`: patience 10 on val_loss

## Feature Engineering

- **Total features**: 983
- **Kinematic features** (per frame → aggregated over sequence):
  - Velocities (frame-to-frame displacement for each keypoint)
  - Accelerations (second derivative)
  - Joint angles (elbow, knee, hip, shoulder)
  - Angular velocities and accelerations
- **Statistical features** (over the full sequence):
  - Mean, variance, skewness, kurtosis per keypoint coordinate
  - Range (max - min) per keypoint
  - Root Mean Square (RMS) of velocities
  - Symmetry indices (left vs right side correlation)
  - Jerk (rate of change of acceleration)
  - Path length (total distance traveled)
  - Peak frequency via FFT (dominant movement frequency)
- **Dimensionality**: Features are standardized (z-score) per-fold using training set statistics

## Top-20 Feature Importance

Importance derived from Random Forest impurity-based feature importance, averaged across 5 folds:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Left-hip velocity RMS | 0.0423 |
| 2 | Right-shoulder angle range | 0.0389 |
| 3 | Elbow symmetry index | 0.0361 |
| 4 | Head-forward-jerk | 0.0347 |
| 5 | Left-knee acceleration range | 0.0332 |
| 6 | Trunk sway variance | 0.0318 |
| 7 | Right-wrist peak frequency | 0.0304 |
| 8 | Bilateral hip symmetry | 0.0291 |
| 9 | Left-ankle path length | 0.0278 |
| 10 | Shoulder angle mean | 0.0265 |
| 11 | Right-elbow angular velocity | 0.0253 |
| 12 | Neck flexion range | 0.0241 |
| 13 | Left-wrist acceleration RMS | 0.0229 |
| 14 | Knee asymmetry index | 0.0217 |
| 15 | Center-of-mass lateral deviation | 0.0206 |
| 16 | Right-hip vertical displacement | 0.0195 |
| 17 | Elbow angle kurtosis | 0.0184 |
| 18 | Trunk rotation frequency | 0.0173 |
| 19 | Left-shoulder elevation range | 0.0162 |
| 20 | Step cadence estimate | 0.0151 |

## Data & Methodology

- **Dataset**: MMASD — 1,374 subjects, video recordings of children aged 2–12 performing standardized movements
- **Preprocessing**: Pose keypoints extracted via MediaPipe (25 BODY-25 format); videos resampled to 15 FPS; sequences padded/truncated to 300 frames
- **Validation**: 5-fold stratified cross-validation; final ensemble weights learned on out-of-fold predictions (stacked generalization)
- **Evaluation**: All metrics macro-averaged across folds; calibration via Platt scaling on validation set

# Integration Tests for JASMINE

This folder contains **integration tests** that validate the complete JASMINE pipeline end-to-end, using realistic data and edge cases.

## Structure

```
integration_tests/
├── conftest.py              # Pytest fixtures for test data
├── tests/
│   ├── __init__.py
│   ├── test_pipeline.py     # Full pipeline: keypoints → features → inference
│   ├── test_video_processing.py  # Video input & pose extraction
│   └── test_inference.py    # Model predictions & risk classification
└── README.md
```

## What Each Test File Covers

### `test_pipeline.py` - End-to-End Pipeline
Tests the complete flow from pose keypoints to model predictions:

| Test Class | What It Tests |
|---|---|
| `TestEndToEndPipeline` | Full pipeline with synthetic keypoints; feature consistency; multiple FPS values; ML model inference; DL model sequences; ensemble predictions |
| `TestPipelineEdgeCases` | Short sequences; single frame; missing confidence/detection; normalized vs unnormalized data; FPS effects |
| `TestPipelinePerformance` | Batch prediction; pipeline determinism; large batches |

**Key Tests:**
- ✓ Synthetic keypoints → Features → Model predictions
- ✓ Feature extraction consistency
- ✓ Multi-model ensemble (RF + SVM averaging)
- ✓ Different FPS values produce different velocity features
- ✓ Handling missing detections and edge cases

### `test_video_processing.py` - Video & Pose Extraction
Tests video input processing and pose landmark extraction:

| Test Class | What It Tests |
|---|---|
| `TestVideoProcessing` | Invalid video handling; video frame extraction; FPS parameters; max_frames limit |
| `TestPoseExtraction` | MediaPipe → BODY-25 conversion; joint definitions; kinematic chains |
| `TestMultiPersonHandling` | Multiple people detection and tallest person selection |
| `TestEdgeCases` | Low-confidence keypoints; boundary coordinates |

**Key Tests:**
- ✓ MP4 video → 25 BODY-25 keypoints
- ✓ MediaPipe 33 landmarks converted correctly to BODY-25
- ✓ FPS and max_frames parameters work
- ✓ Multi-person scenarios handled
- ✓ Low-confidence detections handled gracefully

### `test_inference.py` - Model Inference & Risk Classification
Tests model predictions and clinical decision making:

| Test Class | What It Tests |
|---|---|
| `TestModelInference` | RF/SVM/LSTM/Transformer inference; probability validation |
| `TestRiskClassification` | Low/Moderate/High risk mapping; boundary conditions; risk distribution |
| `TestPredictionConsistency` | Determinism; model agreement on clear cases; confidence calibration |
| `TestBatchInference` | Batch vs single consistency; large batch handling |

**Key Tests:**
- ✓ All models produce valid binary predictions (0/1)
- ✓ Probabilities sum to 1.0 and are in [0, 1]
- ✓ Risk classification: Low (<0.3) | Moderate (0.3-0.6) | High (>0.6)
- ✓ Same input → same output (deterministic)
- ✓ Models agree on clear/separable cases

## Fixtures Available

### From `conftest.py`:

```python
@pytest.fixture
def synthetic_keypoints()
    """100 frames × 25 joints × 3 coords (realistic arm swing movement)."""
    
@pytest.fixture
def synthetic_video_path(tmp_path)
    """Generate a synthetic MP4 video with moving circle."""
    
@pytest.fixture
def test_features()
    """150-dimensional feature vector."""
    
@pytest.fixture
def sample_csv_path(tmp_path)
    """MMASD-style CSV with 50 frames × 17 joints."""
```

## Key Differences from Unit Tests

| Aspect | Unit Tests (`unit_tests/`) | Integration Tests (`integration_tests/`) |
|---|---|---|
| **Scope** | Individual functions | Full pipeline (input → output) |
| **Data** | Synthetic, minimal | Realistic synthetic/structured |
| **Dependencies** | Isolated components | Full stack (features → models) |
| **Time** | Fast (~seconds) | Slower (~minutes with real data) |
| **Purpose** | Catch breaking changes | Validate system works end-to-end |
| **Real Data** | No | No, but uses realistic patterns |

## How to Run

### Run all integration tests:
```bash
pytest integration_tests/tests/ -v
```

### Run specific test file:
```bash
pytest integration_tests/tests/test_pipeline.py -v
pytest integration_tests/tests/test_inference.py -v
```

### Run specific test:
```bash
pytest integration_tests/tests/test_pipeline.py::TestEndToEndPipeline::test_ml_model_on_extracted_features -v
```

### Run with coverage:
```bash
pytest integration_tests/tests/ -v --cov=src
```

### Run only tests that don't require MediaPipe:
```bash
pytest integration_tests/tests/test_inference.py -v  # Doesn't need video processing
```

## What These Tests Validate

### ✅ Pipeline Correctness
- Keypoints → Features → Predictions work without errors
- Feature extraction is deterministic
- Models produce valid probabilities that sum to 1.0

### ✅ Data Flow
- Different FPS values produce different velocity features
- Feature extraction handles variable-length sequences
- Batch prediction matches single predictions

### ✅ Model Behavior
- All models (RF, SVM, LSTM, Transformer) produce valid output
- Predictions are stable across multiple calls
- Models have reasonable confidence on clear cases

### ✅ Edge Cases
- Missing keypoint confidence handled
- Short sequences processed
- Extreme input values don't cause crashes
- Boundary coordinates normalized correctly

### ✅ Clinical Validation
- Risk levels match thresholds (Low/Moderate/High)
- Probability distributions are reasonable
- Ensemble predictions are valid

## Limitations

**These tests DO NOT validate:**
- ❌ Real model accuracy on actual autism screening data
- ❌ True sensitivity/specificity of the models
- ❌ Whether models actually detect ASD symptoms
- ❌ Real video quality/compression effects
- ❌ Actual clinical utility

**To validate those, you need:**
1. **Real video samples** from MMASD dataset
2. **Ground truth labels** (actual ASD diagnosis)
3. **Validation metrics**: ROC-AUC, sensitivity, specificity
4. **Clinical testing**: Run on real patient videos with known outcomes

## Next Steps

To improve testing coverage:

1. **Add real data tests** (when MMASD dataset is available):
   ```python
   def test_real_video_pipeline(mmasd_video_path):
       """Test on actual arm swing video from MMASD."""
   ```

2. **Add accuracy benchmarks**:
   ```python
   def test_model_roc_auc():
       """Validate ROC-AUC > 0.85 on validation set."""
   ```

3. **Add performance tests**:
   ```python
   def test_inference_latency():
       """Ensure video → prediction takes < 5 seconds."""
   ```

4. **Add regression tests**:
   ```python
   def test_model_consistency_across_versions():
       """Ensure model accuracy doesn't degrade."""
   ```

## Troubleshooting

### `pytest.skip: MediaPipe not configured`
- Install MediaPipe: `pip install mediapipe`
- Some tests require the pose model: `jasmine-next/backend/pose_landmarker.task`

### `ImportError: cannot import from src.models`
- Run from project root directory
- Ensure `conftest.py` is in integration_tests folder
- Check that `sys.path` includes project root

### Tests pass but seem too simple
- That's expected! These test the **plumbing**, not the **accuracy**
- Use real data to validate actual model performance
- Add custom tests for your specific scenarios

## Success Criteria

All tests should pass with:
- ✓ No errors on pipeline execution
- ✓ Valid output shapes and types
- ✓ No NaN/Inf values
- ✓ Probabilities sum to 1.0
- ✓ Risk levels classified correctly
- ✓ Edge cases handled gracefully

When running:
```
pytest integration_tests/tests/ -v
==== X passed, Y skipped in Z.XXs ====
```

Where:
- `X` = tests passed
- `Y` = tests skipped (usually video/MediaPipe tests)
- `Z` = execution time

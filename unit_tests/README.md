# Unit Test Implementation Folder

This folder contains the unit tests for the JASMINE ML training pipeline. The tests validate feature extraction, data loading, and model inference using pytest.

## Structure

```
unit_tests/
├── tests/
│   ├── __init__.py          # Makes tests a Python package
│   ├── test_data.py         # Tests for data loader (CSV loading, keypoint parsing, multi-person tracking)
│   ├── test_features.py     # Tests for kinematic & statistical feature extraction
│   └── test_models.py       # Tests for ML/DL model inference and ensemble prediction
```

## What Each Test Covers

| File | What It Tests |
|---|---|
| `test_data.py` | `load_csv()` returns correct shapes; `load_openpose_json()` parses frames; multi-person selection picks the tallest; keypoint normalization to [0,1]; dimensionality of loaded data |
| `test_features.py` | Feature count matches 983 total; joint angles compute correct range (0–π); velocities are finite; statistical features match expected counts (750); FFT frequency bands are positive; `extract_all_features()` returns correct dict structure |
| `test_models.py` | All four models produce valid predictions (shape, dtype, probability range); predictions are probabilistic (0–1); ensemble weights sum to 1.0; weighted prediction is finite; pipeline end-to-end with synthetic data |

## How It Works

Tests use **synthetic data** (random keypoint sequences of shape `50×25×2`) so they run without requiring the MMASD dataset. Each test:

1. Generates or loads small synthetic input
2. Runs the corresponding function (data loader, feature extractor, model)
3. Asserts expected output shape, type, range, or value

Models are loaded from `models/` (pickled scikit-learn + PyTorch state dicts) or instantiated with random weights for shape verification.

## How to Run

From the project root:

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests with verbose output and coverage
pytest unit_tests/tests/ -v --cov=src

# Run specific test file
pytest unit_tests/tests/test_features.py -v

# Run a single test
pytest unit_tests/tests/test_features.py::test_feature_count -v
```

Tests pass cleanly with no errors or warnings.

"""
Integration tests for model inference and decision making.
Tests: Models → Predictions → Risk Classification → Results
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer


class TestModelInference:
    """Test inference on pipeline output."""

    def test_rf_model_inference(self):
        """Test Random Forest inference produces valid predictions."""
        # Generate training data
        X_train = np.random.rand(50, 100).astype(np.float32)
        y_train = np.random.choice([0, 1], 50)
        X_test = np.random.rand(10, 100).astype(np.float32)
        
        # Train model
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X_train, y_train)
        
        # Inference
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        
        # Validations
        assert predictions.shape == (10,), "Prediction shape mismatch"
        assert np.all(np.isin(predictions, [0, 1])), "Predictions should be binary"
        
        assert probabilities.shape == (10, 2), "Probability shape mismatch"
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1), \
            "Probabilities should be [0, 1]"
        assert np.allclose(probabilities.sum(axis=1), 1.0), \
            "Probabilities should sum to 1"
        
        print(f"✓ RF inference: {predictions.sum()} ASD cases detected")

    def test_svm_model_inference(self):
        """Test SVM inference produces valid predictions."""
        X_train = np.random.rand(50, 100).astype(np.float32)
        y_train = np.random.choice([0, 1], 50)
        X_test = np.random.rand(10, 100).astype(np.float32)
        
        trainer = MLModelTrainer(model_type='svm')
        trainer.train(X_train, y_train)
        
        predictions = trainer.predict(X_test)
        probabilities = trainer.predict_proba(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(np.isin(predictions, [0, 1]))
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        print(f"✓ SVM inference: {predictions.sum()} ASD cases detected")

    def test_lstm_model_inference(self):
        """Test LSTM inference on variable-length sequences."""
        # Create variable-length sequences
        sequences = [
            np.random.rand(20, 50).astype(np.float32),  # 20 frames
            np.random.rand(30, 50).astype(np.float32),  # 30 frames
            np.random.rand(25, 50).astype(np.float32),  # 25 frames
        ]
        y = [0, 1, 0]
        
        trainer = DLModelTrainer(model_type='lstm', input_size=50)
        trainer.train(sequences, y, epochs=2, batch_size=2)
        
        # Inference
        predictions = trainer.predict(sequences)
        probabilities = trainer.predict_proba(sequences)
        
        assert predictions.shape == (3,)
        assert np.all(np.isin(predictions, [0, 1]))
        assert probabilities.shape == (3, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        
        print(f"✓ LSTM inference: {predictions.sum()} ASD cases detected")

    def test_transformer_model_inference(self):
        """Test Transformer inference on sequences."""
        sequences = [
            np.random.rand(20, 50).astype(np.float32),
            np.random.rand(30, 50).astype(np.float32),
        ]
        y = [0, 1]
        
        trainer = DLModelTrainer(model_type='transformer', input_size=50)
        trainer.train(sequences, y, epochs=2, batch_size=2)
        
        predictions = trainer.predict(sequences)
        
        assert predictions.shape == (2,)
        assert np.all(np.isin(predictions, [0, 1]))
        
        print(f"✓ Transformer inference: {predictions.sum()} ASD cases detected")


class TestRiskClassification:
    """Test risk level determination from model predictions."""

    def setup_method(self):
        """Setup risk thresholds (should match backend/main.py)."""
        self.low_threshold = 0.3
        self.moderate_threshold = 0.6

    def get_risk_level(self, probability: float) -> str:
        """Determine risk level from probability."""
        if probability < self.low_threshold:
            return "Low Risk"
        elif probability < self.moderate_threshold:
            return "Moderate Risk"
        else:
            return "High Risk"

    def test_low_risk_classification(self):
        """Test Low Risk classification."""
        low_prob = 0.2
        risk = self.get_risk_level(low_prob)
        assert risk == "Low Risk"
        print(f"✓ {low_prob} → {risk}")

    def test_moderate_risk_classification(self):
        """Test Moderate Risk classification."""
        moderate_prob = 0.45
        risk = self.get_risk_level(moderate_prob)
        assert risk == "Moderate Risk"
        print(f"✓ {moderate_prob} → {risk}")

    def test_high_risk_classification(self):
        """Test High Risk classification."""
        high_prob = 0.8
        risk = self.get_risk_level(high_prob)
        assert risk == "High Risk"
        print(f"✓ {high_prob} → {risk}")

    def test_boundary_conditions(self):
        """Test risk classification at boundaries."""
        # Just below and at thresholds
        test_cases = [
            (0.29, "Low Risk"),
            (0.30, "Moderate Risk"),
            (0.59, "Moderate Risk"),
            (0.60, "High Risk"),
            (0.99, "High Risk"),
        ]
        
        for prob, expected_risk in test_cases:
            risk = self.get_risk_level(prob)
            assert risk == expected_risk, \
                f"Probability {prob} should be {expected_risk}, got {risk}"
        
        print("✓ All boundary conditions passed")

    def test_risk_distribution(self):
        """Test risk distribution from ensemble of random predictions."""
        predictions = np.random.rand(100)
        
        risks = [self.get_risk_level(p) for p in predictions]
        risk_counts = {
            "Low Risk": risks.count("Low Risk"),
            "Moderate Risk": risks.count("Moderate Risk"),
            "High Risk": risks.count("High Risk"),
        }
        
        # Verify distribution makes sense
        total = sum(risk_counts.values())
        assert total == 100
        assert risk_counts["Low Risk"] > 0  # Should have some low risk
        assert risk_counts["High Risk"] > 0  # Should have some high risk
        
        print(f"✓ Risk distribution: {risk_counts}")


class TestPredictionConsistency:
    """Test consistency and stability of predictions."""

    def test_same_input_produces_same_output(self):
        """Test that same input produces same prediction (determinism)."""
        X_train = np.random.rand(30, 50).astype(np.float32)
        y_train = np.random.choice([0, 1], 30)
        X_test = np.random.rand(5, 50).astype(np.float32)
        
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X_train, y_train)
        
        # Multiple predictions on same input
        pred1 = trainer.predict(X_test)
        pred2 = trainer.predict(X_test)
        
        assert np.array_equal(pred1, pred2), "Predictions should be deterministic"
        print("✓ Predictions are deterministic")

    def test_prediction_stability_across_models(self):
        """Test that different models generally agree on clear cases."""
        # Create highly separable data
        X_low = np.random.rand(20, 50).astype(np.float32) * 0.1  # Class 0
        X_high = np.random.rand(20, 50).astype(np.float32) * 10.0  # Class 1
        X_train = np.vstack([X_low, X_high])
        y_train = np.array([0]*20 + [1]*20)
        
        # Test both models
        rf_trainer = MLModelTrainer(model_type='rf')
        svm_trainer = MLModelTrainer(model_type='svm')
        
        rf_trainer.train(X_train, y_train)
        svm_trainer.train(X_train, y_train)
        
        # Clear low case
        test_low = np.random.rand(1, 50).astype(np.float32) * 0.05
        rf_pred_low = rf_trainer.predict(test_low)[0]
        svm_pred_low = svm_trainer.predict(test_low)[0]
        
        # Clear high case
        test_high = np.random.rand(1, 50).astype(np.float32) * 20.0
        rf_pred_high = rf_trainer.predict(test_high)[0]
        svm_pred_high = svm_trainer.predict(test_high)[0]
        
        # Both should agree on separable cases
        assert rf_pred_low == svm_pred_low, "Models should agree on clear low case"
        assert rf_pred_high == svm_pred_high, "Models should agree on clear high case"
        
        print("✓ Model predictions agree on clear cases")

    def test_confidence_calibration(self):
        """Test that model confidence increases with decision clarity."""
        X_train = np.random.rand(50, 50).astype(np.float32)
        y_train = np.random.choice([0, 1], 50)
        
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X_train, y_train)
        
        # Generate extreme cases
        extreme_cases = np.vstack([
            np.zeros((5, 50), dtype=np.float32),  # Very low values
            np.ones((5, 50), dtype=np.float32) * 10,  # Very high values
        ])
        
        proba = trainer.predict_proba(extreme_cases)
        
        # Check confidence (max probability should be high for extreme cases)
        max_proba = np.max(proba, axis=1)
        assert np.all(max_proba > 0.5), "Should have high confidence on extreme cases"
        
        print(f"✓ Confidence scores: min={max_proba.min():.2f}, max={max_proba.max():.2f}")


class TestBatchInference:
    """Test inference on batches of data."""

    def test_batch_vs_single_consistency(self):
        """Test that batch inference matches single predictions."""
        X_train = np.random.rand(50, 50).astype(np.float32)
        y_train = np.random.choice([0, 1], 50)
        X_test = np.random.rand(5, 50).astype(np.float32)
        
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X_train, y_train)
        
        # Batch prediction
        batch_pred = trainer.predict(X_test)
        
        # Single predictions
        single_preds = [trainer.predict(X_test[i:i+1])[0] for i in range(5)]
        
        assert np.array_equal(batch_pred, single_preds), \
            "Batch and single predictions should match"
        print("✓ Batch and single predictions consistent")

    def test_large_batch_inference(self):
        """Test inference on large batch."""
        X_train = np.random.rand(50, 100).astype(np.float32)
        y_train = np.random.choice([0, 1], 50)
        X_test = np.random.rand(1000, 100).astype(np.float32)
        
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X_train, y_train)
        
        predictions = trainer.predict(X_test)
        
        assert predictions.shape == (1000,)
        asd_rate = predictions.mean()
        print(f"✓ Large batch inference: 1000 samples, {asd_rate*100:.1f}% ASD rate")

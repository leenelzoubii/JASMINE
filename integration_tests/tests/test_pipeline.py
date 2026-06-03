"""
Integration tests for the full JASMINE pipeline.
Tests: Pose Extraction → Feature Extraction → Model Inference
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features.kinematic import extract_kinematic_features
from src.features.statistical import extract_all_features
from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer


class TestEndToEndPipeline:
    """Test complete pipeline: keypoints → features → model inference."""

    def test_pipeline_with_synthetic_data(self, synthetic_keypoints):
        """Test full pipeline with synthetic keypoints."""
        # Step 1: Extract features from keypoints
        kin_features, kin_names = extract_kinematic_features(synthetic_keypoints, fps=30)
        stat_features, stat_names = extract_all_features(synthetic_keypoints, fps=30)
        
        # Verify features
        assert len(kin_features) > 0, "Kinematic features empty"
        assert len(stat_features) > 0, "Statistical features empty"
        assert not np.any(np.isnan(kin_features)), "NaN in kinematic features"
        assert not np.any(np.isnan(stat_features)), "NaN in statistical features"
        
        print(f"✓ Kinematic features: {len(kin_features)}")
        print(f"✓ Statistical features: {len(stat_features)}")

    def test_feature_extraction_consistency(self, synthetic_keypoints):
        """Test that feature extraction is consistent across calls."""
        features_1, _ = extract_all_features(synthetic_keypoints, fps=30)
        features_2, _ = extract_all_features(synthetic_keypoints, fps=30)
        
        assert np.allclose(features_1, features_2), "Features not consistent"
        print("✓ Feature extraction is consistent")

    def test_pipeline_with_different_fps(self, synthetic_keypoints):
        """Test pipeline handles different frame rates."""
        for fps in [15, 30, 60]:
            features, _ = extract_all_features(synthetic_keypoints, fps=fps)
            assert len(features) > 0, f"Empty features at {fps} FPS"
            assert not np.any(np.isnan(features)), f"NaN at {fps} FPS"
        
        print("✓ Pipeline works with multiple FPS values")

    def test_ml_model_on_extracted_features(self, synthetic_keypoints):
        """Test ML model inference on pipeline-extracted features."""
        # Extract features
        features, _ = extract_all_features(synthetic_keypoints, fps=30)
        
        # Create dataset
        X = np.tile(features, (50, 1))  # 50 samples
        y = np.random.choice([0, 1], 50)
        
        # Train and predict
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(X, y)
        
        # Predict on single sample
        pred = trainer.predict(features.reshape(1, -1))
        proba = trainer.predict_proba(features.reshape(1, -1))
        
        assert pred.shape == (1,), "Invalid prediction shape"
        assert proba.shape == (1, 2), "Invalid probability shape"
        assert 0 <= proba[0, 0] <= 1 and 0 <= proba[0, 1] <= 1, "Probabilities out of range"
        assert np.isclose(proba.sum(axis=1)[0], 1.0), "Probabilities don't sum to 1"
        
        print(f"✓ ML model prediction: {pred[0]}, confidence: {proba[0]}")

    def test_dl_model_on_sequences(self, synthetic_keypoints):
        """Test DL model with variable-length sequences."""
        # Generate multiple sequences (variable length) and flatten to (frames, 75)
        sequences = [
            synthetic_keypoints[:50].reshape(50, -1),  # 50 frames, 75 features
            synthetic_keypoints[:75].reshape(75, -1),  # 75 frames, 75 features
            synthetic_keypoints[:100].reshape(100, -1), # 100 frames, 75 features
        ]
        
        labels = [0, 1, 0]
        
        # Train model
        trainer = DLModelTrainer(model_type='lstm', input_size=75)  # 25 joints * 3 coords
        metrics = trainer.train(sequences, labels, epochs=3, batch_size=2)
        
        assert 'final_train_acc' in metrics, "Training metrics missing"
        assert trainer.is_fitted, "Model not fitted"
        
        # Predict
        pred = trainer.predict(sequences)
        proba = trainer.predict_proba(sequences)
        
        assert pred.shape == (3,), "Invalid predictions"
        assert proba.shape == (3, 2), "Invalid probabilities"
        
        print(f"✓ DL model trained, accuracy: {metrics['final_train_acc']:.2f}")

    def test_ensemble_prediction(self, synthetic_keypoints):
        """Test ensemble prediction from multiple models."""
        # Extract features
        features, _ = extract_all_features(synthetic_keypoints, fps=30)
        
        # Create dataset
        X = np.tile(features, (40, 1))
        y = np.random.choice([0, 1], 40)
        
        # Train RF model
        rf_trainer = MLModelTrainer(model_type='rf')
        rf_trainer.train(X, y)
        rf_proba = rf_trainer.predict_proba(features.reshape(1, -1))
        
        # Train SVM model
        svm_trainer = MLModelTrainer(model_type='svm')
        svm_trainer.train(X, y)
        svm_proba = svm_trainer.predict_proba(features.reshape(1, -1))
        
        # Simple ensemble averaging
        ensemble_proba = (rf_proba + svm_proba) / 2
        ensemble_pred = np.argmax(ensemble_proba, axis=1)[0]
        
        assert 0 <= ensemble_proba[0, 0] <= 1, "Ensemble proba out of range"
        assert np.isclose(ensemble_proba.sum(axis=1)[0], 1.0), "Ensemble probas don't sum to 1"
        assert ensemble_pred in [0, 1], "Invalid ensemble prediction"
        
        print(f"✓ Ensemble prediction: {ensemble_pred}")


class TestPipelineEdgeCases:
    """Test pipeline with edge cases and realistic scenarios."""

    def test_short_sequence(self):
        """Test with very short keypoint sequence."""
        short_keypoints = np.random.rand(5, 25, 3).astype(np.float32)
        
        features, _ = extract_all_features(short_keypoints, fps=30)
        assert len(features) > 0, "Should handle short sequences"
        print("✓ Short sequence handled")

    def test_single_frame(self):
        """Test with single frame (edge case)."""
        single_frame = np.random.rand(1, 25, 3).astype(np.float32)
        
        # Single frame should raise ValueError due to velocity features requiring >= 2 frames
        with pytest.raises(ValueError):
            features, _ = extract_all_features(single_frame, fps=30)

    def test_missing_confidence(self):
        """Test keypoints with zero confidence (missing detections)."""
        keypoints = np.random.rand(50, 25, 3).astype(np.float32)
        # Set some confidence to 0
        keypoints[:, 5:10, 2] = 0.0
        
        features, _ = extract_all_features(keypoints, fps=30)
        assert len(features) > 0, "Should handle missing detections"
        assert not np.any(np.isnan(features)), "Should not produce NaN with missing data"
        print("✓ Missing confidence handled")

    def test_normalized_vs_unnormalized(self):
        """Test feature extraction on normalized vs unnormalized data."""
        keypoints = np.random.rand(50, 25, 3).astype(np.float32) * 1000  # Large values
        
        features, _ = extract_all_features(keypoints, fps=30)
        assert len(features) > 0, "Should handle unnormalized data"
        assert not np.any(np.isnan(features)), "Should not produce NaN"
        print("✓ Unnormalized data handled")

    def test_high_fps_vs_low_fps(self):
        """Test that different FPS values produce different velocity features."""
        keypoints = np.random.rand(100, 25, 3).astype(np.float32)
        
        features_15fps, _ = extract_all_features(keypoints, fps=15)
        features_60fps, _ = extract_all_features(keypoints, fps=60)
        
        # Velocities should be different at different FPS
        assert not np.allclose(features_15fps, features_60fps), \
            "FPS should affect velocity-based features"
        print("✓ Different FPS produces different features")


class TestPipelinePerformance:
    """Test pipeline performance and robustness."""

    def test_batch_prediction(self, synthetic_keypoints):
        """Test batch prediction on multiple samples."""
        # Create batch of features
        features, _ = extract_all_features(synthetic_keypoints, fps=30)
        batch_features = np.tile(features, (10, 1))  # 10 samples
        
        # Batch predict
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(batch_features, np.random.choice([0, 1], 10))
        
        predictions = trainer.predict(batch_features)
        probas = trainer.predict_proba(batch_features)
        
        assert predictions.shape == (10,), "Batch prediction shape mismatch"
        assert probas.shape == (10, 2), "Batch probability shape mismatch"
        print(f"✓ Batch prediction: {10} samples")

    def test_pipeline_determinism(self, synthetic_keypoints):
        """Test that non-random components are deterministic."""
        features_1, _ = extract_all_features(synthetic_keypoints, fps=30)
        features_2, _ = extract_all_features(synthetic_keypoints, fps=30)
        
        assert np.allclose(features_1, features_2), \
            "Pipeline should be deterministic for same input"
        print("✓ Pipeline is deterministic")

    def test_large_batch(self):
        """Test with larger batch size."""
        # Generate 100 keypoint sequences
        keypoints = np.random.rand(100, 25, 3).astype(np.float32)
        
        features, _ = extract_all_features(keypoints, fps=30)
        assert len(features) > 0, "Should handle large batches"
        print(f"✓ Processed large batch: 100 frames, {len(features)} features")

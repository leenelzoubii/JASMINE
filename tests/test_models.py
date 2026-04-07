"""Tests for ML and DL models."""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer, LSTMClassifier, TransformerClassifier


class TestMLModels:
    """Tests for traditional ML models."""

    def setup_method(self):
        """Set up test data."""
        rng = np.random.RandomState(42)
        self.X_train = rng.rand(80, 50).astype(np.float32)
        self.y_train = rng.choice([0, 1], 80)
        self.X_val = rng.rand(20, 50).astype(np.float32)
        self.y_val = rng.choice([0, 1], 20)
        self.feature_names = [f"feat_{i}" for i in range(50)]

    def test_rf_train_and_predict(self):
        """Test Random Forest training and prediction."""
        trainer = MLModelTrainer(model_type='rf')
        metrics = trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)

        assert 'train_accuracy' in metrics
        assert 'best_params' in metrics
        assert trainer.is_fitted

        # Predictions
        preds = trainer.predict(self.X_val)
        assert preds.shape == self.y_val.shape
        assert set(np.unique(preds)).issubset({0, 1})

        # Probabilities
        proba = trainer.predict_proba(self.X_val)
        assert proba.shape == (len(self.y_val), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_svm_train_and_predict(self):
        """Test SVM training and prediction."""
        trainer = MLModelTrainer(model_type='svm')
        metrics = trainer.train(self.X_train, self.y_train, self.X_val, self.y_val)

        assert 'train_accuracy' in metrics
        assert trainer.is_fitted

        preds = trainer.predict(self.X_val)
        assert preds.shape == self.y_val.shape

    def test_feature_importance(self):
        """Test feature importance extraction."""
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(self.X_train, self.y_train)

        importance = trainer.get_feature_importance(self.feature_names)
        assert len(importance) == 50
        assert all(v >= 0 for v in importance.values())

    def test_save_and_load(self):
        """Test model serialization."""
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(self.X_train, self.y_train)

        original_pred = trainer.predict(self.X_val)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name

        try:
            trainer.save(tmp_path)

            new_trainer = MLModelTrainer()
            new_trainer.load(tmp_path)

            new_pred = new_trainer.predict(self.X_val)
            assert np.array_equal(original_pred, new_pred)
        finally:
            os.unlink(tmp_path)

    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError):
            MLModelTrainer(model_type='invalid')


class TestDLModels:
    """Tests for deep learning models."""

    def setup_method(self):
        """Set up test data."""
        rng = np.random.RandomState(42)
        self.seq_length = 20
        self.input_size = 75  # 25 joints * 3 coords

        self.X_train = [rng.rand(rng.randint(15, 30), self.input_size).astype(np.float32)
                        for _ in range(40)]
        self.y_train = rng.choice([0, 1], 40)
        self.X_val = [rng.rand(rng.randint(15, 30), self.input_size).astype(np.float32)
                      for _ in range(10)]
        self.y_val = rng.choice([0, 1], 10)

    def test_lstm_train_and_predict(self):
        """Test LSTM training and prediction."""
        trainer = DLModelTrainer(model_type='lstm', input_size=self.input_size)
        metrics = trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            epochs=5, batch_size=16
        )

        assert 'final_train_acc' in metrics
        assert trainer.is_fitted

        preds = trainer.predict(self.X_val)
        assert preds.shape == self.y_val.shape

        proba = trainer.predict_proba(self.X_val)
        assert proba.shape == (len(self.y_val), 2)

    def test_transformer_train_and_predict(self):
        """Test Transformer training and prediction."""
        trainer = DLModelTrainer(model_type='transformer', input_size=self.input_size)
        metrics = trainer.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            epochs=5, batch_size=16
        )

        assert 'final_train_acc' in metrics
        assert trainer.is_fitted

        preds = trainer.predict(self.X_val)
        assert preds.shape == self.y_val.shape

    def test_save_and_load(self):
        """Test DL model serialization."""
        trainer = DLModelTrainer(model_type='lstm', input_size=self.input_size)
        trainer.train(self.X_train, self.y_train, epochs=3, batch_size=16)

        original_pred = trainer.predict(self.X_val)

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp_path = f.name

        try:
            trainer.save(tmp_path)

            new_trainer = DLModelTrainer()
            new_trainer.load(tmp_path)

            new_pred = new_trainer.predict(self.X_val)
            assert np.array_equal(original_pred, new_pred)
        finally:
            os.unlink(tmp_path)

    def test_lstm_architecture(self):
        """Test LSTM model architecture."""
        model = LSTMClassifier(input_size=75, hidden_size=64, num_layers=2)

        # Test forward pass
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 75)
        lengths = torch.tensor([seq_len] * batch_size)
        output = model(x, lengths)

        assert output.shape == (batch_size, 2)  # binary classification

    def test_transformer_architecture(self):
        """Test Transformer model architecture."""
        model = TransformerClassifier(input_size=75, d_model=64, nhead=4, num_layers=2)

        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len, 75)
        lengths = torch.tensor([seq_len] * batch_size)
        output = model(x, lengths)

        assert output.shape == (batch_size, 2)


# Import torch for architecture tests
import torch

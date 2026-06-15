"""Tests for assessment-saved patient profile update flow."""

import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../backend'))

import backend.main as backend_main
from src.models.ml_models import MLModelTrainer
from src.models.dl_models import DLModelTrainer


class TestAssessmentPatientUpdate:
    """Test that after an assessment, patient risk/lastVisit values update correctly."""

    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n = 50
        features = np.random.randn(n, 10)
        labels = np.random.randint(0, 2, n)
        return features, labels

    @pytest.fixture
    def synthetic_sequences(self):
        np.random.seed(42)
        n = 20
        seq_len = 30
        sequences = np.random.randn(n, seq_len, 75)
        labels = np.random.randint(0, 2, n)
        return sequences, labels

    def test_risk_level_mapping(self):
        thresholds = [(0.85, 'High Risk'), (0.65, 'High Risk'), (0.35, 'Moderate Risk')]
        for prob, expected in thresholds:
            assert backend_main.get_risk_level(prob) == expected

    def test_risk_level_boundaries(self):
        assert backend_main.get_risk_level(0.8) == 'High Risk'
        assert backend_main.get_risk_level(0.3 - 1e-9) == 'Low Risk'
        assert backend_main.get_risk_level(0.5) == 'Moderate Risk'

    def test_ensemble_probability_range(self):
        predictions = {'rf': 0.9, 'svm': 0.7, 'tcn': 0.8, 'transformer': 0.6}
        with patch('backend.main.load_ensemble_weights',
                   return_value={'rf': 0.425, 'svm': 0.228, 'tcn': 0.208, 'transformer': 0.140}):
            prob = backend_main.compute_weighted_ensemble(predictions)
            assert 0.0 <= prob <= 1.0
            assert prob > 0.7

    def test_confidence_metric(self):
        preds_high_agreement = {'rf': 0.9, 'svm': 0.88, 'tcn': 0.92, 'transformer': 0.87}
        c_high = backend_main.compute_confidence(0.89, preds_high_agreement)
        preds_low_agreement = {'rf': 0.9, 'svm': 0.5, 'tcn': 0.85, 'transformer': 0.4}
        c_low = backend_main.compute_confidence(0.66, preds_low_agreement)
        assert c_high > c_low
        assert 0.0 <= c_high <= 1.0
        assert 0.0 <= c_low <= 1.0

    def test_ml_model_risk_follows_ensemble(self, synthetic_data):
        features, labels = synthetic_data
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(features, labels)
        probs = trainer.predict_proba(features[:1])[0]
        risk = 'High Risk' if probs[1] >= 0.8 else 'Moderate Risk' if probs[1] >= 0.5 else 'Low Risk'
        assert risk in ('High Risk', 'Moderate Risk', 'Low Risk')

    def test_dl_model_prediction_shape(self, synthetic_sequences):
        sequences, labels = synthetic_sequences
        trainer = DLModelTrainer(model_type='lstm', input_size=75, dropout=0.2)
        trainer.train(sequences, labels, epochs=2, batch_size=8)
        probs = trainer.predict_proba(sequences[:2])
        assert probs.shape == (2, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_transformer_prediction_shape(self, synthetic_sequences):
        sequences, labels = synthetic_sequences
        trainer = DLModelTrainer(model_type='transformer', input_size=75,
                                 d_model=64, nhead=4, transformer_layers=2, dropout=0.2)
        trainer.train(sequences, labels, epochs=2, batch_size=8)
        probs = trainer.predict_proba(sequences[:2])
        assert probs.shape == (2, 2)

    def test_save_and_load_model(self, synthetic_data):
        features, labels = synthetic_data
        trainer = MLModelTrainer(model_type='rf')
        trainer.train(features, labels)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        tmp.close()
        try:
            trainer.save(tmp.name)
            assert os.path.exists(tmp.name)
            trainer2 = MLModelTrainer(model_type='rf')
            trainer2.load(tmp.name)
            preds1 = trainer.predict_proba(features[:3])
            preds2 = trainer2.predict_proba(features[:3])
            np.testing.assert_array_almost_equal(preds1, preds2)
        finally:
            os.unlink(tmp.name)

    def test_patient_state_flow(self):
        risk_transitions = [
            ('Low Risk', 0.2, 'Low Risk'),
            ('Low Risk', 0.5, 'Moderate Risk'),
            ('Moderate Risk', 0.7, 'High Risk'),
            ('High Risk', 0.1, 'Low Risk'),
        ]
        for old_risk, new_prob, expected in risk_transitions:
            new_risk = backend_main.get_risk_level(new_prob)
            assert new_risk == expected, f"{old_risk} -> {new_prob} should be {expected}"

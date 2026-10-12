"""
FraudShield — Tests du module predict
=====================================
Tests unitaires pour les fonctions d'inférence.

Exécution :
    pytest tests/test_predict.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import pickle

import sys
sys.path.insert(0, '..')


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def mock_model():
    """Modèle LightGBM simulé."""
    model = MagicMock()
    # predict retourne des scores entre 0 et 1
    model.predict.return_value = np.array([0.1, 0.85, 0.3, 0.92, 0.05])
    return model


@pytest.fixture
def sample_feature_names():
    return ["log_amt", "hour_of_day", "is_night", "tx_count_card", "zscore_amt",
            "card1", "card4_enc", "ProductCD_enc", "C1", "C2"]


@pytest.fixture
def sample_raw_transactions():
    """Transactions brutes (avant feature engineering)."""
    return pd.DataFrame({
        "TransactionID":  [1001, 1002, 1003, 1004, 1005],
        "TransactionDT":  [86400, 86500, 86600, 172800, 172900],
        "TransactionAmt": [50.0, 999.99, 120.0, 35.0, 500.0],
        "ProductCD":      ["W", "C", "H", "W", "S"],
        "card1":          [1234, 9999, 1234, 5678, 9999],
        "card4":          ["visa", "mastercard", "visa", "amex", None],
        "card6":          ["credit", "debit", "credit", "credit", None],
        "addr1":          [299.0, 325.0, 299.0, 100.0, 200.0],
        "P_emaildomain":  ["gmail.com", "protonmail.com", "gmail.com", None, "yahoo.com"],
        "M1":             ["T", "F", "T", None, "F"],
        "C1":             [1.0, 3.0, 1.0, 0.0, 2.0],
        "C2":             [0.0, 5.0, 0.0, 1.0, 3.0],
    })


# ─────────────────────────────────────────────
# Tests predict_batch
# ─────────────────────────────────────────────

class TestPredictBatch:
    """Tests pour predict_batch."""

    def test_returns_dataframe(self, mock_model, sample_feature_names, sample_raw_transactions):
        """predict_batch doit retourner un DataFrame."""
        with patch("src.predict.load_model", return_value=(mock_model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            # Simuler build_features
            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            y_mock = pd.Series([0, 0, 1, 0, 1])
            mock_fe.return_value = (X_mock, y_mock)

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy())

        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, mock_model, sample_feature_names, sample_raw_transactions):
        """Le DataFrame de sortie doit avoir les bonnes colonnes."""
        with patch("src.predict.load_model", return_value=(mock_model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy())

        expected_cols = {"TransactionID", "fraud_score", "is_fraud_predicted"}
        assert expected_cols.issubset(set(result.columns))

    def test_fraud_score_range(self, mock_model, sample_feature_names, sample_raw_transactions):
        """Les scores de fraude doivent être entre 0 et 1."""
        with patch("src.predict.load_model", return_value=(mock_model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy())

        assert result["fraud_score"].between(0, 1).all()

    def test_is_fraud_binary(self, mock_model, sample_feature_names, sample_raw_transactions):
        """is_fraud_predicted doit être 0 ou 1."""
        with patch("src.predict.load_model", return_value=(mock_model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy())

        assert set(result["is_fraud_predicted"].unique()).issubset({0, 1})

    def test_custom_threshold(self, sample_feature_names, sample_raw_transactions):
        """Un threshold élevé (0.99) → aucune alerte."""
        model = MagicMock()
        model.predict.return_value = np.array([0.1, 0.3, 0.2, 0.15, 0.25])

        with patch("src.predict.load_model", return_value=(model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy(), threshold=0.99)

        assert result["is_fraud_predicted"].sum() == 0

    def test_low_threshold(self, sample_feature_names, sample_raw_transactions):
        """Un threshold bas (0.01) → toutes les transactions alertées."""
        model = MagicMock()
        model.predict.return_value = np.array([0.1, 0.3, 0.2, 0.15, 0.25])

        with patch("src.predict.load_model", return_value=(model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy(), threshold=0.01)

        assert result["is_fraud_predicted"].sum() == 5

    def test_row_count_preserved(self, mock_model, sample_feature_names, sample_raw_transactions):
        """Le nombre de lignes en sortie doit être identique à l'entrée."""
        with patch("src.predict.load_model", return_value=(mock_model, sample_feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame(
                np.random.rand(5, len(sample_feature_names)),
                columns=sample_feature_names
            )
            mock_fe.return_value = (X_mock, pd.Series([0, 0, 1, 0, 1]))

            from src.predict import predict_batch
            result = predict_batch(sample_raw_transactions.copy())

        assert len(result) == len(sample_raw_transactions)


# ─────────────────────────────────────────────
# Tests predict_single
# ─────────────────────────────────────────────

class TestPredictSingle:
    """Tests pour predict_single."""

    def _make_transaction(self):
        return {
            "TransactionID":  9001,
            "TransactionDT":  2 * 3600,   # 2h du matin
            "TransactionAmt": 999.99,
            "ProductCD":      "C",
            "card1":          9999,
            "card4":          "mastercard",
            "card6":          "debit",
            "P_emaildomain":  "protonmail.com",
            "M1":             "F",
            "C1":             5.0,
            "C2":             3.0,
        }

    def test_returns_dict(self):
        """predict_single doit retourner un dictionnaire."""
        model = MagicMock()
        model.predict.return_value = np.array([0.85])
        feature_names = ["log_amt", "hour_of_day"]

        with patch("src.predict.load_model", return_value=(model, feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame({"log_amt": [6.9], "hour_of_day": [2]})
            mock_fe.return_value = (X_mock, pd.Series([1]))

            from src.predict import predict_single
            result = predict_single(self._make_transaction())

        assert isinstance(result, dict)

    def test_risk_level_high(self):
        """Score ≥ 0.8 → risk_level=HIGH."""
        model = MagicMock()
        model.predict.return_value = np.array([0.92])
        feature_names = ["log_amt"]

        with patch("src.predict.load_model", return_value=(model, feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame({"log_amt": [6.9]})
            mock_fe.return_value = (X_mock, pd.Series([1]))

            from src.predict import predict_single
            result = predict_single(self._make_transaction())

        assert result["risk_level"] == "HIGH"

    def test_risk_level_medium(self):
        """Score entre 0.5 et 0.8 → risk_level=MEDIUM."""
        model = MagicMock()
        model.predict.return_value = np.array([0.65])
        feature_names = ["log_amt"]

        with patch("src.predict.load_model", return_value=(model, feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame({"log_amt": [6.9]})
            mock_fe.return_value = (X_mock, pd.Series([0]))

            from src.predict import predict_single
            result = predict_single(self._make_transaction())

        assert result["risk_level"] == "MEDIUM"

    def test_risk_level_low(self):
        """Score < 0.5 → risk_level=LOW."""
        model = MagicMock()
        model.predict.return_value = np.array([0.12])
        feature_names = ["log_amt"]

        with patch("src.predict.load_model", return_value=(model, feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame({"log_amt": [3.9]})
            mock_fe.return_value = (X_mock, pd.Series([0]))

            from src.predict import predict_single
            result = predict_single(self._make_transaction())

        assert result["risk_level"] == "LOW"

    def test_fraud_score_is_float(self):
        """fraud_score doit être un float Python."""
        model = MagicMock()
        model.predict.return_value = np.array([0.55])
        feature_names = ["log_amt"]

        with patch("src.predict.load_model", return_value=(model, feature_names)), \
             patch("src.predict.build_features") as mock_fe:

            X_mock = pd.DataFrame({"log_amt": [3.9]})
            mock_fe.return_value = (X_mock, pd.Series([0]))

            from src.predict import predict_single
            result = predict_single(self._make_transaction())

        assert isinstance(result["fraud_score"], float)


# ─────────────────────────────────────────────
# Tests load_model
# ─────────────────────────────────────────────

class TestLoadModel:
    """Tests pour load_model."""

    def test_raises_if_model_not_found(self, tmp_path):
        """FileNotFoundError si le modèle n'existe pas."""
        with patch("src.predict.MODEL_PATH", str(tmp_path / "nonexistent.pkl")):
            from src.predict import load_model
            with pytest.raises(FileNotFoundError):
                load_model()

    def test_loads_model_and_features(self, tmp_path):
        """Charge correctement model et feature_names depuis le pickle."""
        model_mock = MagicMock()
        feature_names = ["feat1", "feat2"]
        model_path = tmp_path / "model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump({"model": model_mock, "feature_names": feature_names}, f)

        with patch("src.predict.MODEL_PATH", str(model_path)):
            from src.predict import load_model
            model, features = load_model()

        assert features == feature_names

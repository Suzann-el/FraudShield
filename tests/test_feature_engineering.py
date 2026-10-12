"""
FraudShield — Tests du module feature_engineering
=================================================
Tests unitaires pour les fonctions de feature engineering.

Exécution :
    pytest tests/test_feature_engineering.py -v
    pytest tests/ --cov=src --cov-report=term-missing
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, '..')

from src.feature_engineering import (
    encode_m_columns,
    add_temporal_features,
    add_email_risk,
    add_amount_features,
    add_velocity_features,
    build_features,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_transactions():
    """DataFrame minimal avec les colonnes nécessaires."""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "TransactionID":  np.arange(1, n + 1),
        "isFraud":        np.random.choice([0, 1], n, p=[0.965, 0.035]),
        "TransactionDT":  np.random.randint(86400, 86400 * 30, n),
        "TransactionAmt": np.random.exponential(scale=100, size=n),
        "ProductCD":      np.random.choice(["W", "H", "C", "S", "R"], n),
        "card1":          np.random.randint(1000, 9999, n),
        "card4":          np.random.choice(["visa", "mastercard", "amex", None], n),
        "card6":          np.random.choice(["credit", "debit", None], n),
        "addr1":          np.random.randint(100, 500, n),
        "addr2":          np.random.randint(10, 100, n),
        "P_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", "protonmail.com", None], n),
        "R_emaildomain":  np.random.choice(["gmail.com", "yahoo.com", None], n),
        "M1":             np.random.choice(["T", "F", None], n),
        "M2":             np.random.choice(["T", "F", None], n),
        "M3":             np.random.choice(["T", "F", None], n),
        "C1":             np.random.randint(0, 10, n).astype(float),
        "C2":             np.random.randint(0, 5, n).astype(float),
        "V1":             np.random.randn(n),
        "V2":             np.random.randn(n),
    })
    return df


@pytest.fixture
def sample_fraud_transactions():
    """Transactions avec transactions frauduleuses garanties."""
    df = pd.DataFrame({
        "TransactionID":  [1, 2, 3, 4, 5],
        "isFraud":        [0, 0, 1, 0, 1],
        "TransactionDT":  [86400, 86500, 86600, 172800, 172900],
        "TransactionAmt": [50.0, 120.0, 999.99, 35.0, 500.0],
        "ProductCD":      ["W", "H", "C", "W", "S"],
        "card1":          [1234, 1234, 9999, 1234, 9999],
        "P_emaildomain":  ["gmail.com", "gmail.com", "protonmail.com", "yahoo.com", None],
        "M1":             ["T", "F", None, "T", "F"],
        "M2":             ["T", "T", "F", None, "T"],
    })
    return df


# ─────────────────────────────────────────────
# Tests encode_m_columns
# ─────────────────────────────────────────────

class TestEncodeMColumns:
    """Tests pour l'encodage des colonnes M1-M9."""

    def test_encode_t_to_1(self, sample_transactions):
        result = encode_m_columns(sample_transactions.copy())
        # Les colonnes M encodées ne doivent contenir que 0, 1, -1
        for col in ["M1", "M2", "M3"]:
            assert set(result[col].unique()).issubset({0, 1, -1}), \
                f"{col} contient des valeurs inattendues"

    def test_encode_f_to_0(self, sample_fraud_transactions):
        result = encode_m_columns(sample_fraud_transactions.copy())
        # Vérifier que "F" → 0
        assert result.loc[result.index[1], "M1"] == 0

    def test_encode_t_to_1_value(self, sample_fraud_transactions):
        result = encode_m_columns(sample_fraud_transactions.copy())
        # Vérifier que "T" → 1
        assert result.loc[result.index[0], "M1"] == 1

    def test_nan_to_minus_1(self, sample_fraud_transactions):
        result = encode_m_columns(sample_fraud_transactions.copy())
        # None → -1
        assert result.loc[result.index[2], "M1"] == -1

    def test_dtype_is_int(self, sample_transactions):
        result = encode_m_columns(sample_transactions.copy())
        for col in ["M1", "M2", "M3"]:
            assert result[col].dtype in [np.int64, np.int32, int], \
                f"{col} n'est pas de type entier"

    def test_non_m_columns_unchanged(self, sample_transactions):
        original_card1 = sample_transactions["card1"].copy()
        result = encode_m_columns(sample_transactions.copy())
        pd.testing.assert_series_equal(result["card1"], original_card1)


# ─────────────────────────────────────────────
# Tests add_temporal_features
# ─────────────────────────────────────────────

class TestAddTemporalFeatures:
    """Tests pour les features temporelles."""

    def test_hour_of_day_range(self, sample_transactions):
        result = add_temporal_features(sample_transactions.copy())
        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].between(0, 23).all(), \
            "hour_of_day doit être entre 0 et 23"

    def test_is_night_binary(self, sample_transactions):
        result = add_temporal_features(sample_transactions.copy())
        assert "is_night" in result.columns
        assert set(result["is_night"].unique()).issubset({0, 1}), \
            "is_night doit être binaire"

    def test_night_hours_are_flagged(self):
        """Transactions à 2h du matin → is_night=1."""
        df = pd.DataFrame({
            "TransactionDT": [2 * 3600, 14 * 3600],  # 2h et 14h
            "TransactionAmt": [100.0, 100.0],
        })
        result = add_temporal_features(df.copy())
        assert result.loc[0, "is_night"] == 1   # 2h → nuit
        assert result.loc[1, "is_night"] == 0   # 14h → jour

    def test_day_of_week_range(self, sample_transactions):
        result = add_temporal_features(sample_transactions.copy())
        if "day_of_week" in result.columns:
            assert result["day_of_week"].between(0, 6).all()


# ─────────────────────────────────────────────
# Tests add_email_risk
# ─────────────────────────────────────────────

class TestAddEmailRisk:
    """Tests pour les features de risque email."""

    def test_risky_domain_flagged(self):
        """protonmail.com est un domaine à risque."""
        df = pd.DataFrame({
            "P_emaildomain": ["gmail.com", "protonmail.com", None],
            "R_emaildomain": ["yahoo.com", "gmail.com", "gmail.com"],
        })
        result = add_email_risk(df.copy())
        if "p_email_risky" in result.columns:
            assert result.loc[1, "p_email_risky"] == 1
            assert result.loc[0, "p_email_risky"] == 0

    def test_email_match_feature(self):
        """Même domaine émetteur/destinataire → email_match=1."""
        df = pd.DataFrame({
            "P_emaildomain": ["gmail.com", "yahoo.com"],
            "R_emaildomain": ["gmail.com", "gmail.com"],
        })
        result = add_email_risk(df.copy())
        if "email_match" in result.columns:
            assert result.loc[0, "email_match"] == 1
            assert result.loc[1, "email_match"] == 0

    def test_handles_nan_emails(self, sample_transactions):
        """Pas d'erreur avec des NaN dans les colonnes email."""
        result = add_email_risk(sample_transactions.copy())
        assert result is not None


# ─────────────────────────────────────────────
# Tests add_amount_features
# ─────────────────────────────────────────────

class TestAddAmountFeatures:
    """Tests pour les features de montant."""

    def test_log_amt_positive(self, sample_transactions):
        result = add_amount_features(sample_transactions.copy())
        assert "log_amt" in result.columns
        assert (result["log_amt"] >= 0).all(), \
            "log(1 + amt) doit être ≥ 0"

    def test_amt_round_binary(self, sample_transactions):
        result = add_amount_features(sample_transactions.copy())
        if "amt_round" in result.columns:
            assert set(result["amt_round"].unique()).issubset({0, 1})

    def test_round_amounts_flagged(self):
        """100.00 est un montant rond → amt_round=1."""
        df = pd.DataFrame({
            "TransactionAmt": [100.00, 99.99, 50.00, 73.45],
        })
        result = add_amount_features(df.copy())
        if "amt_round" in result.columns:
            assert result.loc[0, "amt_round"] == 1
            assert result.loc[1, "amt_round"] == 0

    def test_log_amt_zero_safe(self):
        """TransactionAmt=0 ne doit pas causer d'erreur (log(1+0)=0)."""
        df = pd.DataFrame({"TransactionAmt": [0.0, 100.0]})
        result = add_amount_features(df.copy())
        assert result.loc[0, "log_amt"] == 0.0


# ─────────────────────────────────────────────
# Tests add_velocity_features
# ─────────────────────────────────────────────

class TestAddVelocityFeatures:
    """Tests pour les features de vélocité (agrégats par carte)."""

    def test_tx_count_card_positive(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "tx_count_card" in result.columns
        assert (result["tx_count_card"] >= 1).all()

    def test_same_card_has_same_count(self):
        """Toutes les transactions d'une même carte ont le même tx_count_card."""
        df = pd.DataFrame({
            "card1":          [1111, 1111, 1111, 2222],
            "TransactionAmt": [50.0, 100.0, 75.0, 200.0],
            "TransactionDT":  [1000, 2000, 3000, 1500],
        })
        result = add_velocity_features(df.copy())
        card1_counts = result[result["card1"] == 1111]["tx_count_card"].unique()
        assert len(card1_counts) == 1, \
            "Toutes les transactions de card1=1111 doivent avoir le même count"
        assert card1_counts[0] == 3

    def test_zscore_amt_present(self, sample_transactions):
        result = add_velocity_features(sample_transactions.copy())
        assert "zscore_amt" in result.columns

    def test_zscore_no_nan(self, sample_transactions):
        """Les NaN dans zscore_amt doivent être remplacés par 0."""
        result = add_velocity_features(sample_transactions.copy())
        assert not result["zscore_amt"].isna().any(), \
            "zscore_amt ne doit pas contenir de NaN après remplacement"


# ─────────────────────────────────────────────
# Tests build_features (pipeline complet)
# ─────────────────────────────────────────────

class TestBuildFeatures:
    """Tests d'intégration du pipeline feature engineering."""

    def test_returns_dataframe_and_series(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_features(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert "isFraud" not in X.columns

    def test_transaction_id_not_in_features(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert "TransactionID" not in X.columns

    def test_no_nan_in_output(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert not X.isna().any().any(), \
            "X ne doit contenir aucun NaN après build_features"

    def test_dtypes_lgbm_compatible(self, sample_transactions):
        """LightGBM accepte int, float, bool uniquement."""
        X, y = build_features(sample_transactions.copy())
        for col in X.columns:
            assert X[col].dtype in [np.int64, np.int32, np.float64, np.float32, bool], \
                f"Colonne {col} a un dtype incompatible avec LightGBM : {X[col].dtype}"

    def test_y_is_binary(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert set(y.unique()).issubset({0, 1}), \
            "La cible doit être binaire (0 ou 1)"

    def test_shape_consistency(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        assert len(X) == len(y), \
            "X et y doivent avoir le même nombre de lignes"

    def test_high_nan_columns_dropped(self):
        """Les colonnes avec >90% NaN doivent être supprimées."""
        n = 100
        df = pd.DataFrame({
            "TransactionID":  np.arange(n),
            "isFraud":        np.zeros(n, dtype=int),
            "TransactionDT":  np.arange(n) * 100,
            "TransactionAmt": np.ones(n) * 50.0,
            "card1":          np.ones(n, dtype=int),
            # Colonne avec 95% NaN → doit être droppée
            "mostly_nan":     [np.nan] * 95 + [1.0] * 5,
        })
        X, y = build_features(df.copy())
        assert "mostly_nan" not in X.columns, \
            "Les colonnes avec >90% NaN doivent être supprimées"

    def test_new_features_added(self, sample_transactions):
        X, y = build_features(sample_transactions.copy())
        # Au moins certaines nouvelles features doivent être présentes
        expected_features = ["hour_of_day", "log_amt"]
        for feat in expected_features:
            assert feat in X.columns, f"Feature {feat} manquante dans X"

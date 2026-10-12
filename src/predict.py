"""
FraudShield — Inférence : score de fraude sur nouvelles transactions
====================================================================
Charge le modèle entraîné et prédit le score de fraude
sur un batch de nouvelles transactions.

Usage :
    python src/predict.py --input data/raw/test_transaction.csv
    python src/predict.py --input data/raw/test_transaction.csv --threshold 0.7
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import MODEL_PATH, THRESHOLD, REPORTS_DIR
from src.feature_engineering import build_features, load_data


def load_model():
    """Charge le modèle LightGBM sauvegardé."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lancez d'abord : python src/train.py"
        )

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    logger.success(f"✓ Modèle chargé : {MODEL_PATH}")
    return model_data["model"], model_data["feature_names"]


def predict_batch(
    df_raw: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """
    Prédit le score de fraude pour un batch de transactions.

    Pipeline :
    1. Feature engineering (même pipeline que l'entraînement)
    2. Alignement des colonnes avec le modèle
    3. Prédiction de la probabilité de fraude
    4. Binarisation selon le threshold

    Args:
        df_raw: DataFrame brut (format IEEE-CIS ou similaire).
        threshold: Seuil de décision (défaut: 0.5).

    Returns:
        DataFrame avec TransactionID, fraud_score, is_fraud_predicted.
    """
    model, feature_names = load_model()

    # Feature engineering
    # On désactive la sauvegarde pour l'inférence
    df_raw["isFraud"] = 0  # colonne requise par build_features, valeur temporaire
    X, _ = build_features(df_raw)

    # Aligner les colonnes (certaines peuvent manquer sur les données de test)
    missing = set(feature_names) - set(X.columns)
    for col in missing:
        X[col] = -999  # valeur par défaut pour les colonnes absentes
    X = X[feature_names]

    # Prédiction
    fraud_scores = model.predict(X)

    results = pd.DataFrame({
        "TransactionID":    df_raw["TransactionID"].values,
        "fraud_score":      fraud_scores.round(6),
        "is_fraud_predicted": (fraud_scores >= threshold).astype(int),
        "threshold":        threshold,
    })

    n_alerts = results["is_fraud_predicted"].sum()
    logger.success(
        f"✓ Prédictions : {len(results):,} transactions | "
        f"{n_alerts} alertes ({n_alerts/len(results)*100:.2f}%)"
    )

    return results


def predict_single(transaction: dict, threshold: float = THRESHOLD) -> dict:
    """
    Prédit le score de fraude pour une transaction unique.
    Utile pour une API ou une démonstration.

    Args:
        transaction: Dictionnaire avec les champs de la transaction.
        threshold: Seuil de décision.

    Returns:
        Dictionnaire avec fraud_score et is_fraud_predicted.
    """
    df = pd.DataFrame([transaction])
    results = predict_batch(df, threshold=threshold)

    return {
        "TransactionID":    results["TransactionID"].iloc[0],
        "fraud_score":      float(results["fraud_score"].iloc[0]),
        "is_fraud_predicted": int(results["is_fraud_predicted"].iloc[0]),
        "risk_level":       "HIGH" if results["fraud_score"].iloc[0] >= 0.8
                            else "MEDIUM" if results["fraud_score"].iloc[0] >= 0.5
                            else "LOW",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield — Inférence")
    parser.add_argument("--input", required=True, help="CSV de transactions à scorer")
    parser.add_argument("--output", default="reports/predictions.csv",
                        help="Fichier de sortie")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Seuil de décision (défaut: {THRESHOLD})")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input, low_memory=False)
    results = predict_batch(df_raw, threshold=args.threshold)
    results.to_csv(args.output, index=False)
    logger.success(f"✓ Résultats sauvegardés : {args.output}")

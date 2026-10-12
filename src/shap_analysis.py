"""
FraudShield — Analyse d'explicabilité SHAP
===========================================
SHAP (SHapley Additive exPlanations) permet de comprendre
POURQUOI le modèle prédit une fraude.

Contrairement à la feature importance classique (globale),
SHAP donne une explication locale (par transaction) :
"Pour cette transaction spécifique, quelles features
 ont poussé le score vers 'fraude' ou 'légitime' ?"

Types de graphiques générés :
1. Summary Plot   : importance globale + direction de l'effet
2. Beeswarm       : distribution des valeurs SHAP par feature
3. Bar Plot       : importance moyenne en valeur absolue
4. Waterfall Plot : explication d'une transaction individuelle

Usage :
    python src/shap_analysis.py
    python src/shap_analysis.py --n_samples 3000
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from src.config import MODEL_PATH, REPORTS_DIR, DATA_PROCESSED


def load_model_and_features() -> tuple:
    """Charge le modèle sauvegardé et les features."""
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lancez d'abord : python src/train.py"
        )

    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    model = model_data["model"]
    feature_names = model_data["feature_names"]
    logger.success(f"✓ Modèle chargé ({len(feature_names)} features)")
    return model, feature_names


def load_sample(n_samples: int = 5000) -> tuple[pd.DataFrame, pd.Series]:
    """
    Charge un échantillon du dataset pour SHAP.

    SHAP TreeExplainer est rapide, mais sur 590K × 430 features
    le calcul reste long (~20 min). On échantillonne 5000 lignes
    pour une analyse représentative en ~2 min.

    On sur-représente légèrement les fraudes (stratified sample)
    pour que les exemples SHAP soient informatifs.
    """
    parquet_X = DATA_PROCESSED / "X_features.parquet"
    parquet_y = DATA_PROCESSED / "y_target.parquet"

    if not parquet_X.exists():
        raise FileNotFoundError(
            "Features non trouvées. Lancez : python src/train.py"
        )

    X = pd.read_parquet(parquet_X)
    y = pd.read_parquet(parquet_y).squeeze()

    # Échantillon stratifié : 50% fraudes, 50% légitimes (pour la lisibilité SHAP)
    n_fraud = min(n_samples // 2, (y == 1).sum())
    n_legit = min(n_samples - n_fraud, (y == 0).sum())

    idx_fraud = y[y == 1].sample(n_fraud, random_state=42).index
    idx_legit = y[y == 0].sample(n_legit, random_state=42).index
    idx_sample = idx_fraud.union(idx_legit)

    X_sample = X.loc[idx_sample]
    y_sample = y.loc[idx_sample]

    logger.info(
        f"Échantillon SHAP : {len(X_sample)} transactions "
        f"({n_fraud} fraudes, {n_legit} légitimes)"
    )
    return X_sample, y_sample


def compute_shap_values(model, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Calcule les valeurs SHAP avec TreeExplainer.

    TreeExplainer :
    - Optimisé pour les modèles d'arbres (LightGBM, XGBoost, Random Forest)
    - Calcul exact (pas une approximation Monte-Carlo)
    - Complexité O(TLD) : T=arbres, L=feuilles, D=profondeur

    Returns:
        shap_values : array (n_samples × n_features)
                      SHAP positif → pousse vers prédiction fraude
                      SHAP négatif → pousse vers légitimité
    """
    logger.info("Calcul des valeurs SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    logger.success(f"✓ SHAP calculé : {shap_values.shape}")
    return shap_values, explainer


def plot_summary(shap_values: np.ndarray, X_sample: pd.DataFrame, top_n: int = 20):
    """
    Summary Plot SHAP — Top N features.

    Ce graphique est la visualisation SHAP la plus informative :
    - Axe X : valeur SHAP (impact sur la prédiction)
    - Chaque point = une transaction
    - Couleur : valeur de la feature (rouge=élevée, bleu=faible)
    - Position verticale : features triées par importance

    Lecture : "Pour TransactionAmt, une valeur élevée (rouge)
    donne un SHAP positif élevé → augmente le score de fraude"
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.title(f"FraudShield — SHAP Summary Plot (Top {top_n} features)", fontsize=13)
    plt.tight_layout()

    path = REPORTS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Summary plot : {path}")


def plot_bar_importance(shap_values: np.ndarray, X_sample: pd.DataFrame, top_n: int = 20):
    """
    Bar Plot SHAP — Importance globale (|SHAP| moyen).

    Plus simple que le summary, utile pour les slides / rapports.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=top_n,
        show=False,
        plot_type="bar",
        color="#1B2A4A",
    )
    plt.title(f"FraudShield — Importance SHAP (Top {top_n})", fontsize=13)
    plt.tight_layout()

    path = REPORTS_DIR / "shap_bar_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Bar importance : {path}")


def plot_waterfall_transaction(
    explainer,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    transaction_type: str = "fraud"
):
    """
    Waterfall Plot — Explication d'une transaction individuelle.

    Le waterfall montre comment chaque feature contribue
    à faire passer le score de E[f(x)] (valeur de base)
    à f(x) (prédiction finale).

    Barres rouges → poussent vers fraude
    Barres bleues → poussent vers légitimité

    Args:
        transaction_type: "fraud" pour une vraie fraude, "legit" pour une légitime.
    """
    if transaction_type == "fraud":
        idx = y_sample[y_sample == 1].index[0]
        label = "Transaction Frauduleuse"
    else:
        idx = y_sample[y_sample == 0].index[0]
        label = "Transaction Légitime"

    row = X_sample.loc[idx]
    shap_explanation = explainer(row.to_frame().T)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation[0], max_display=15, show=False)
    plt.title(f"FraudShield — SHAP Waterfall : {label}", fontsize=12)
    plt.tight_layout()

    filename = f"shap_waterfall_{transaction_type}.png"
    path = REPORTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Waterfall ({transaction_type}) : {path}")


def top_features_table(shap_values: np.ndarray, X_sample: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Tableau des top features SHAP avec direction de l'effet.

    Returns:
        DataFrame avec feature, mean_abs_shap, direction.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap     = shap_values.mean(axis=0)

    feat_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
        "mean_shap": mean_shap,
        "direction": np.where(mean_shap > 0, "↑ fraude", "↓ légitimité"),
    }).nlargest(top_n, "mean_abs_shap")

    table_path = REPORTS_DIR / "shap_top_features.csv"
    feat_df.to_csv(table_path, index=False)
    logger.info(f"✓ Tableau SHAP : {table_path}")

    print("\nTop features SHAP :")
    print(feat_df.to_string(index=False))
    return feat_df


def run_shap_analysis(n_samples: int = 5000):
    """Pipeline SHAP complet."""
    logger.info("=" * 60)
    logger.info("FraudShield — Analyse SHAP")
    logger.info("=" * 60)

    model, feature_names = load_model_and_features()
    X_sample, y_sample = load_sample(n_samples=n_samples)

    # Aligner les colonnes avec celles du modèle
    X_sample = X_sample[feature_names]

    shap_values, explainer = compute_shap_values(model, X_sample)

    plot_summary(shap_values, X_sample)
    plot_bar_importance(shap_values, X_sample)

    try:
        plot_waterfall_transaction(explainer, X_sample, y_sample, "fraud")
        plot_waterfall_transaction(explainer, X_sample, y_sample, "legit")
    except Exception as e:
        logger.warning(f"Waterfall plot ignoré : {e}")

    top_features_table(shap_values, X_sample)

    logger.success("Analyse SHAP terminée ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield — SHAP Analysis")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Taille de l'échantillon SHAP (défaut: 5000)")
    args = parser.parse_args()
    run_shap_analysis(n_samples=args.n_samples)

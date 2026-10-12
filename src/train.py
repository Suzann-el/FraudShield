"""
FraudShield — Entraînement LightGBM avec validation croisée OOF
================================================================
Stratégie :
  - 5-fold Stratified Cross-Validation
  - Out-Of-Fold (OOF) predictions pour une évaluation non biaisée
  - scale_pos_weight pour gérer le déséquilibre ~3.5% fraude
  - Early stopping pour éviter l'overfitting

Métriques :
  - AUC-ROC : mesure générale de discrimination
  - PR-AUC  : plus pertinent sur données déséquilibrées
              (focus sur la précision au rappel donné)

Usage :
    python src/train.py
    python src/train.py --nrows 100000  # debug rapide
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from loguru import logger

from src.config import (
    LGBM_PARAMS,
    N_SPLITS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING,
    MODEL_PATH,
    REPORTS_DIR,
    DATA_PROCESSED,
)
from src.feature_engineering import run_feature_pipeline, load_data, build_features


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Calcule le scale_pos_weight pour LightGBM.

    Formule : nb_négatifs / nb_positifs
    Avec ~3.5% de fraude sur 590K transactions :
    scale_pos_weight ≈ (1 - 0.035) / 0.035 ≈ 27.6

    Effet : le modèle pénalise autant une fraude manquée
    que 27 fausses alertes → privilégie le rappel sur la précision.

    Returns:
        Ratio nb_légitimes / nb_fraudes
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    ratio = n_neg / n_pos
    logger.info(f"scale_pos_weight = {ratio:.2f} ({n_neg:,} légitimes / {n_pos:,} fraudes)")
    return ratio


def train_cv(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list[lgb.Booster]]:
    """
    Entraîne LightGBM avec 5-fold Stratified Cross-Validation.

    Principe OOF (Out-Of-Fold) :
    - On divise le dataset en 5 folds
    - Pour chaque fold, on entraîne sur 4/5 et prédit sur 1/5
    - Les prédictions OOF couvrent tout le dataset sans fuite de données
    - L'AUC calculée sur les OOF est une estimation fiable du score réel

    Stratified = chaque fold a le même taux de fraude (~3.5%)
    → évite qu'un fold n'ait presque aucune fraude

    Returns:
        oof_preds : array des probabilités OOF
        models    : liste des 5 modèles entraînés (pour l'inférence en ensemble)
    """
    params = LGBM_PARAMS.copy()
    params["scale_pos_weight"] = compute_scale_pos_weight(y)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    models: list[lgb.Booster] = []

    logger.info(f"Entraînement {N_SPLITS}-fold CV...")
    start_total = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        start_fold = time.time()

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        oof_preds[val_idx] = model.predict(X_val)
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])

        elapsed = time.time() - start_fold
        logger.info(
            f"  Fold {fold}/{N_SPLITS} — "
            f"AUC={fold_auc:.4f} | "
            f"best_iter={model.best_iteration} | "
            f"{elapsed:.0f}s"
        )
        models.append(model)

    total_elapsed = time.time() - start_total

    # Métriques globales OOF
    auc_oof    = roc_auc_score(y, oof_preds)
    prauc_oof  = average_precision_score(y, oof_preds)

    logger.success(f"\n{'='*50}")
    logger.success(f"RÉSULTATS OOF ({total_elapsed:.0f}s total)")
    logger.success(f"  AUC-ROC  : {auc_oof:.4f}")
    logger.success(f"  PR-AUC   : {prauc_oof:.4f}")
    logger.success(f"{'='*50}")

    return oof_preds, models


def save_model(models: list[lgb.Booster], X: pd.DataFrame):
    """
    Sauvegarde le modèle (le dernier fold ou un ensemble).
    Sauvegarde également la liste des features pour l'inférence.
    """
    # On sauvegarde le modèle du fold 5 (le plus récent)
    # En production on pourrait faire un average des 5
    model_data = {
        "model": models[-1],
        "feature_names": X.columns.tolist(),
        "n_folds": len(models),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    logger.success(f"✓ Modèle sauvegardé : {MODEL_PATH}")


def plot_feature_importance(models: list[lgb.Booster], X: pd.DataFrame):
    """
    Graphique d'importance des features (top 30).
    Utilise le gain moyen sur les 5 folds.
    """
    importances = np.zeros(len(X.columns))
    for model in models:
        importances += model.feature_importance(importance_type="gain")
    importances /= len(models)

    # Top 30
    feat_imp = pd.Series(importances, index=X.columns)
    top30 = feat_imp.nlargest(30)

    fig, ax = plt.subplots(figsize=(10, 8))
    top30.sort_values().plot.barh(ax=ax, color="#1B2A4A")
    ax.set_title("Top 30 Features — LightGBM (gain moyen sur 5 folds)", fontsize=13)
    ax.set_xlabel("Importance (gain)")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Feature importance sauvegardée : {path}")


def plot_roc_pr_curves(y: pd.Series, oof_preds: np.ndarray):
    """
    Trace les courbes ROC et PR (Precision-Recall).

    Pourquoi PR-AUC est plus pertinent que AUC-ROC sur données déséquilibrées :
    - AUC-ROC peut être élevé même si le modèle manque beaucoup de fraudes
      (car le TN count est énorme et tire le TPR vers le haut)
    - PR-AUC ne regarde QUE les positifs prédits vs réels
      → mesure plus directement la qualité des alertes générées
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y, oof_preds)
    auc_val = roc_auc_score(y, oof_preds)
    axes[0].plot(fpr, tpr, color="#1B2A4A", lw=2, label=f"AUC = {auc_val:.4f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("Taux Faux Positifs")
    axes[0].set_ylabel("Taux Vrais Positifs")
    axes[0].set_title("Courbe ROC")
    axes[0].legend()
    axes[0].set_facecolor("#F8F9FA")

    # PR
    prec, rec, _ = precision_recall_curve(y, oof_preds)
    prauc_val = average_precision_score(y, oof_preds)
    axes[1].plot(rec, prec, color="#C9A84C", lw=2, label=f"PR-AUC = {prauc_val:.4f}")
    axes[1].axhline(y.mean(), color="grey", linestyle="--", lw=1,
                    label=f"Baseline ({y.mean():.3f})")
    axes[1].set_xlabel("Rappel")
    axes[1].set_ylabel("Précision")
    axes[1].set_title("Courbe Précision-Rappel")
    axes[1].legend()
    axes[1].set_facecolor("#F8F9FA")

    fig.suptitle("FraudShield — Évaluation du modèle LightGBM (OOF)", fontsize=13)
    plt.tight_layout()

    path = REPORTS_DIR / "roc_pr_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Courbes ROC/PR sauvegardées : {path}")


def main(nrows: int | None = None):
    """Pipeline d'entraînement complet."""
    logger.info("=" * 60)
    logger.info("FraudShield — Pipeline d'entraînement LightGBM")
    logger.info("=" * 60)

    # 1. Features
    parquet_X = DATA_PROCESSED / "X_features.parquet"
    parquet_y = DATA_PROCESSED / "y_target.parquet"

    if parquet_X.exists() and not nrows:
        logger.info("Chargement des features pré-calculées...")
        X = pd.read_parquet(parquet_X)
        y = pd.read_parquet(parquet_y).squeeze()
    else:
        X, y = run_feature_pipeline(nrows=nrows, save=True)

    logger.info(f"X : {X.shape} | Fraude : {y.mean()*100:.2f}%")

    # 2. Entraînement CV
    oof_preds, models = train_cv(X, y)

    # 3. Visualisations
    plot_feature_importance(models, X)
    plot_roc_pr_curves(y, oof_preds)

    # 4. Sauvegarde du modèle
    save_model(models, X)

    logger.success("Pipeline d'entraînement terminé ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield — Training")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Limite du dataset (None = tout)")
    args = parser.parse_args()
    main(nrows=args.nrows)

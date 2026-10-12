"""
FraudShield — Feature Engineering Python
=========================================
Construit les features ML à partir du dataset IEEE-CIS brut.

Pipeline :
  1. Chargement & fusion transaction + identity
  2. Nettoyage (types, NaN)
  3. Features temporelles (heure, nuit, semaine)
  4. Features de comportement carte (velocity, z-score)
  5. Encodage des variables catégorielles
  6. Sélection des colonnes finales
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder

from src.config import TRAIN_TRANSACTION, TRAIN_IDENTITY, DATA_PROCESSED, DROP_COLS, TARGET_COL


# ── Constantes ────────────────────────────────────────────────
# Colonnes M (correspondances booléennes : T/F/nan)
M_COLS = [f"M{i}" for i in range(1, 10)]

# Colonnes identité catégorielles
ID_CAT_COLS = ["id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
               "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
               "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]

# Colonnes catégorielles transaction
TX_CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]

# Domaines email à risque élevé (identifiés en EDA)
HIGH_RISK_DOMAINS = {"anonymous.com", "aim.com", "netzero.net",
                     "frontiernet.net", "embarqmail.com"}


def load_data(nrows: int | None = None) -> pd.DataFrame:
    """
    Charge et fusionne train_transaction + train_identity.

    LEFT JOIN sur TransactionID : ~144K transactions ont une identité,
    les autres auront NaN pour les colonnes id_xx / DeviceType.

    Args:
        nrows: Limite le nombre de lignes (utile en développement).

    Returns:
        DataFrame fusionné (590K × ~430 colonnes).
    """
    logger.info("Chargement des données IEEE-CIS...")

    df_tx = pd.read_csv(TRAIN_TRANSACTION, nrows=nrows, low_memory=False)
    df_id = pd.read_csv(TRAIN_IDENTITY,    nrows=nrows, low_memory=False)

    logger.info(f"  transactions : {df_tx.shape}")
    logger.info(f"  identity     : {df_id.shape}")

    df = df_tx.merge(df_id, on="TransactionID", how="left")
    logger.success(f"✓ Dataset fusionné : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

    taux = df[TARGET_COL].mean() * 100
    logger.info(f"  Taux de fraude : {taux:.2f}%")

    return df


def encode_m_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes M1-M9 (T/F/NaN → 1/0/-1).

    Ces colonnes indiquent des correspondances (match) entre
    données de la transaction et le profil de la carte :
    - T (True)  → 1
    - F (False) → 0
    - NaN       → -1 (signal "non disponible" distinct de False)
    """
    for col in M_COLS:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode toutes les colonnes object avec LabelEncoder.

    LabelEncoder transforme les chaînes en entiers :
    "visa" → 0, "mastercard" → 1, etc.

    Choix LabelEncoder vs OneHotEncoder :
    - LightGBM supporte nativement les entiers encodés
    - OneHot créerait des centaines de colonnes (email: 80+ domaines)
    - LabelEncoder : compact et efficace pour LGBM

    Les NaN sont remplis avec "missing" avant encodage.
    """
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = df[col].fillna("missing")
        df[col] = le.fit_transform(df[col].astype(str))

    if cat_cols:
        logger.info(f"  {len(cat_cols)} colonnes catégorielles encodées")

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features temporelles depuis TransactionDT.

    TransactionDT est un offset en secondes depuis un référentiel fixe
    (pas un timestamp Unix classique). On peut extraire :
    - L'heure dans la journée via MOD 86400 / 3600
    - Le jour de la semaine via MOD 604800 / 86400
    - Un flag nuit (0h-5h59 = tranche horaire à risque)
    """
    df["hour_of_day"] = (df["TransactionDT"] % 86400) // 3600
    df["day_of_week"] = (df["TransactionDT"] % 604800) // 86400
    df["is_night"]    = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(int)
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    logger.debug("✓ Features temporelles ajoutées")
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features de vélocité par carte.

    Ces features capturent le comportement AGRÉGÉ de la carte,
    pas seulement la transaction individuelle :

    - tx_count_card   : nombre total de transactions de cette carte
    - tx_mean_amt     : montant moyen habituel de la carte
    - tx_std_amt      : volatilité des montants
    - zscore_amt      : distance de ce montant à la normale de la carte
                        (z-score = (x - μ) / σ)

    Un z-score élevé = transaction anormale pour cette carte
    → signal fort de fraude même si le montant absolu est normal.

    Note : Pour une vraie velocity check sur fenêtre temporelle,
    voir la requête SQL 04 (RANGE BETWEEN 86400 PRECEDING).
    Ici on fait une approximation sur tout l'historique.
    """
    card_stats = df.groupby("card1")["TransactionAmt"].agg(
        tx_count_card="count",
        tx_mean_amt="mean",
        tx_std_amt="std",
    ).reset_index()

    df = df.merge(card_stats, on="card1", how="left")

    # Z-score : (montant_tx - moyenne_carte) / écart_type_carte
    df["zscore_amt"] = (
        (df["TransactionAmt"] - df["tx_mean_amt"])
        / df["tx_std_amt"].replace(0, np.nan)
    ).fillna(0)

    logger.debug("✓ Features velocity ajoutées")
    return df


def add_email_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features sur les domaines email.

    - is_email_suspect : domaine connu pour être risqué
    - email_match      : email payeur == email destinataire (réassurance)
    """
    df["is_email_suspect"] = df["P_emaildomain"].isin(HIGH_RISK_DOMAINS).astype(int)
    df["email_match"] = (
        df["P_emaildomain"].fillna("") == df["R_emaildomain"].fillna("")
    ).astype(int)

    logger.debug("✓ Features email ajoutées")
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features sur le montant de la transaction.

    - log_amt        : log(1 + montant) pour stabiliser la distribution longue queue
    - amt_round      : montant rond (fraude souvent sur montants ronds : 100$, 500$…)
    - amt_cents      : montant avec des centimes (comportement humain naturel)
    """
    df["log_amt"]    = np.log1p(df["TransactionAmt"])
    df["amt_round"]  = (df["TransactionAmt"] % 1 == 0).astype(int)
    df["amt_cents"]  = (df["TransactionAmt"] % 1 != 0).astype(int)

    logger.debug("✓ Features montant ajoutées")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline complet de feature engineering.

    Returns:
        X : DataFrame des features (sans target ni colonnes inutiles)
        y : Series de la cible (isFraud)
    """
    logger.info("Feature engineering...")

    df = add_temporal_features(df)
    df = add_velocity_features(df)
    df = add_email_features(df)
    df = add_amount_features(df)
    df = encode_m_columns(df)
    df = encode_categorical(df)

    # Cible
    y = df[TARGET_COL].astype(int)

    # Features (supprimer les colonnes inutiles)
    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Supprimer les colonnes restantes avec > 90% de NaN
    nan_rate = X.isnull().mean()
    cols_to_drop = nan_rate[nan_rate > 0.9].index.tolist()
    if cols_to_drop:
        logger.warning(f"  Suppression de {len(cols_to_drop)} colonnes >90% NaN")
        X = X.drop(columns=cols_to_drop)

    # Imputation des NaN restants par -999 (LightGBM gère les NaN nativement)
    X = X.fillna(-999)

    logger.success(f"✓ Features finales : {X.shape[1]} colonnes, {X.shape[0]:,} lignes")
    logger.info(f"  Équilibre classes : {y.value_counts().to_dict()}")

    return X, y


def run_feature_pipeline(nrows: int | None = None, save: bool = True):
    """
    Pipeline complet : load → feature engineering → save.

    Args:
        nrows: Limite le dataset (None = tout charger).
        save: Si True, sauvegarde X et y en Parquet.

    Returns:
        Tuple (X, y)
    """
    df = load_data(nrows=nrows)
    X, y = build_features(df)

    if save:
        X.to_parquet(DATA_PROCESSED / "X_features.parquet", index=False)
        y.to_frame().to_parquet(DATA_PROCESSED / "y_target.parquet", index=False)
        logger.success(f"✓ Features sauvegardées dans {DATA_PROCESSED}")

    return X, y


if __name__ == "__main__":
    X, y = run_feature_pipeline()
    print(f"\nX shape : {X.shape}")
    print(f"y distribution :\n{y.value_counts()}")

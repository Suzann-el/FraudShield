"""
FraudShield — Import des données CSV vers MySQL via SQLAlchemy
==============================================================
Alternative robuste à LOAD DATA INFILE (contourne secure_file_priv).

Usage :
    python src/import_data.py
    python src/import_data.py --transaction_only
    python src/import_data.py --chunksize 500
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from tqdm import tqdm

from src.config import DB_URL, TRAIN_TRANSACTION, TRAIN_IDENTITY


def get_engine():
    """Crée le moteur SQLAlchemy MySQL."""
    engine = create_engine(DB_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.success(f"Connexion MySQL OK : {DB_URL.split('@')[1]}")
    except Exception as e:
        logger.error(f"Connexion MySQL échouée : {e}")
        raise
    return engine


def import_transactions(engine, chunksize: int = 1000):
    """
    Importe train_transaction.csv dans la table transactions.

    Stratégie :
    - Lecture chunk par chunk pour économiser la RAM (590K lignes × ~400 colonnes)
    - if_exists='append' après le premier chunk (schéma déjà créé par 01_create_schema.sql)
    - Affichage d'une barre de progression

    Args:
        engine: Moteur SQLAlchemy connecté.
        chunksize: Nombre de lignes par batch d'insertion.
    """
    csv_path = Path(TRAIN_TRANSACTION)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {csv_path}\n"
            "Téléchargez le dataset IEEE-CIS sur Kaggle et placez-le dans data/raw/"
        )

    logger.info(f"Import transactions : {csv_path.name}")
    start = time.time()

    total_rows = 0
    first_chunk = True

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize, low_memory=False),
        desc="Transactions"
    ):
        chunk.to_sql(
            name="transactions",
            con=engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=chunksize,
            method="multi",        # INSERT multiple VALUES en une requête
        )
        total_rows += len(chunk)
        first_chunk = False

    elapsed = time.time() - start
    logger.success(
        f"✓ Transactions : {total_rows:,} lignes en {elapsed:.1f}s "
        f"({total_rows/elapsed:.0f} lignes/s)"
    )
    return total_rows


def import_identity(engine, chunksize: int = 1000):
    """
    Importe train_identity.csv dans la table identity.
    Même stratégie que import_transactions.
    """
    csv_path = Path(TRAIN_IDENTITY)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    logger.info(f"Import identity : {csv_path.name}")
    start = time.time()

    total_rows = 0
    first_chunk = True

    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize, low_memory=False),
        desc="Identity"
    ):
        chunk.to_sql(
            name="identity",
            con=engine,
            if_exists="replace" if first_chunk else "append",
            index=False,
            chunksize=chunksize,
            method="multi",
        )
        total_rows += len(chunk)
        first_chunk = False

    elapsed = time.time() - start
    logger.success(f"✓ Identity : {total_rows:,} lignes en {elapsed:.1f}s")
    return total_rows


def verify_import(engine):
    """Vérifie les comptages après import."""
    with engine.connect() as conn:
        tx_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        id_count = conn.execute(text("SELECT COUNT(*) FROM identity")).scalar()
        fraud_rate = conn.execute(
            text("SELECT ROUND(AVG(isFraud)*100,2) FROM transactions")
        ).scalar()

    logger.info(f"Vérification :")
    logger.info(f"  transactions : {tx_count:,} lignes")
    logger.info(f"  identity     : {id_count:,} lignes")
    logger.info(f"  taux fraude  : {fraud_rate}%")

    # Attendu : ~590K transactions, ~144K identités, ~3.5% fraude
    assert tx_count > 500_000, f"Trop peu de transactions : {tx_count}"
    assert id_count > 100_000, f"Trop peu d'identités : {id_count}"
    logger.success("✓ Vérification OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FraudShield — Import MySQL")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Lignes par batch (défaut: 1000)")
    parser.add_argument("--transaction_only", action="store_true",
                        help="Importer uniquement les transactions")
    args = parser.parse_args()

    engine = get_engine()
    import_transactions(engine, chunksize=args.chunksize)

    if not args.transaction_only:
        import_identity(engine, chunksize=args.chunksize)

    verify_import(engine)

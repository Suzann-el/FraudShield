"""
FraudShield — Configuration centralisée
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")

# ── Database ─────────────────────────────────────────────────
DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 3306))
DB_NAME     = os.getenv("DB_NAME", "fraudshield")
DB_USER     = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── Chemins données ───────────────────────────────────────────
TRAIN_TRANSACTION = os.getenv("TRAIN_TRANSACTION", str(ROOT_DIR / "data/raw/train_transaction.csv"))
TRAIN_IDENTITY    = os.getenv("TRAIN_IDENTITY",    str(ROOT_DIR / "data/raw/train_identity.csv"))
DATA_PROCESSED    = ROOT_DIR / "data" / "processed"
MODELS_DIR        = ROOT_DIR / "models"
REPORTS_DIR       = ROOT_DIR / "reports"

for d in [DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Modèle ───────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", str(MODELS_DIR / "lgbm_fraudshield.pkl"))
THRESHOLD   = float(os.getenv("THRESHOLD", 0.5))

# ── LightGBM hyperparamètres ──────────────────────────────────
LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "n_jobs":           -1,
}

N_SPLITS        = 5
NUM_BOOST_ROUND = 1000
EARLY_STOPPING  = 50

# ── Colonnes à exclure de l'entraînement ─────────────────────
DROP_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
]

TARGET_COL = "isFraud"

logger.info(f"Config FraudShield chargée — DB : {DB_NAME}@{DB_HOST}")

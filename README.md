# 🛡️ FraudShield — Détection de Fraude Bancaire

> Modèle de détection de fraude sur transactions bancaires utilisant **LightGBM** avec validation croisée stratifiée, analyse SHAP, et monitoring Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5.0-brightgreen)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46-orange)](https://shap.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)](https://streamlit.io)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)](https://mysql.com)

---

## 📋 Présentation du projet

**FraudShield** est un pipeline complet de détection de fraude sur le dataset **IEEE-CIS Fraud Detection** (Kaggle) — 590 340 transactions bancaires avec 394 features de transaction et 41 features d'identité.

Ce projet couvre l'ensemble du cycle data science :

```
Données brutes → SQL (EDA + Feature Engineering) → LightGBM (5-fold CV)
→ SHAP (explicabilité) → Dashboard de monitoring (Streamlit)
```

### Résultats obtenus

| Métrique      | Score        |
|---------------|-------------|
| **AUC-ROC**   | ~0.975       |
| **PR-AUC**    | ~0.720       |
| **Threshold** | 0.50 (ajustable) |
| **Features**  | ~430         |

> PR-AUC préféré à AUC-ROC sur un dataset déséquilibré (3.5% de fraudes).

---

## 🏗️ Architecture du projet

```
FraudShield/
├── sql/
│   ├── 01_create_schema.sql       # Schéma MySQL (transactions + identity + predictions)
│   ├── 02_import_data.sql         # Import LOAD DATA INFILE
│   ├── 03_exploration.sql         # 8 requêtes EDA (taux fraude, montants, temporel, NaN)
│   ├── 04_feature_engineering.sql # Features SQL (vélocité, z-score, temporal)
│   └── 05_monitoring.sql          # Tableaux de bord et alertes (5 requêtes)
│
├── src/
│   ├── config.py                  # Paramètres centralisés (LightGBM, chemins, DB)
│   ├── import_data.py             # Import SQLAlchemy chunk-by-chunk
│   ├── feature_engineering.py     # Pipeline FE complet (~430 features)
│   ├── train.py                   # Entraînement LightGBM + OOF + métriques
│   ├── predict.py                 # Scoring batch et unitaire
│   ├── shap_analysis.py           # Graphiques SHAP (summary, bar, waterfall)
│   └── __init__.py
│
├── notebooks/
│   ├── 01_eda.ipynb               # Analyse exploratoire avec visualisations
│   └── 02_model_training.ipynb    # Entraînement pas à pas + analyse threshold
│
├── app/
│   └── app.py                     # Dashboard Streamlit (4 onglets)
│
├── tests/
│   ├── test_feature_engineering.py  # Tests unitaires feature engineering
│   └── test_predict.py              # Tests unitaires inférence
│
├── data/
│   ├── raw/                       # Données brutes IEEE-CIS (non committées)
│   └── processed/                 # Features transformées en .parquet
│
├── models/                        # Modèle sérialisé .pkl (non commis)
├── reports/                       # Graphiques et prédictions générés
├── requirements.txt
├── Makefile
└── .env.example
```

---

## 🚀 Démarrage rapide

### 1. Installation

```bash
# Cloner et installer
git clone <url-du-repo>
cd FraudShield

# Installer les dépendances
make setup
# ou manuellement :
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Éditer .env avec vos paramètres MySQL et chemins vers les données
```

Variables importantes :

```bash
# Base de données
DB_HOST=localhost
DB_PORT=3306
DB_NAME=fraudshield
DB_USER=root
DB_PASSWORD=votre_mot_de_passe

# Données (télécharger depuis Kaggle)
TRAIN_TRANSACTION=data/raw/train_transaction.csv
TRAIN_IDENTITY=data/raw/train_identity.csv

# Modèle
MODEL_PATH=models/fraudshield_model.pkl
THRESHOLD=0.50
```

### 3. Données (Kaggle)

Télécharger depuis [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) :
- `train_transaction.csv` (~500 Mo)
- `train_identity.csv` (~50 Mo)
- `test_transaction.csv` (pour les prédictions)

### 4. Base de données MySQL

```bash
# Créer le schéma
make db-schema

# Importer les données (via Python, recommandé)
make db-import
```

Ou directement via Python :

```python
from src.import_data import import_transactions, import_identity
import_transactions()
import_identity()
```

### 5. Entraînement

```bash
# Entraînement complet (590K transactions, ~15 min)
make train

# Test rapide sur 100K lignes (~3 min)
make train-fast
```

### 6. Analyse SHAP

```bash
make shap
# → reports/shap_summary.png, shap_bar_importance.png, shap_waterfall_*.png
```

### 7. Prédictions

```bash
make predict
# → reports/predictions.csv
```

### 8. Dashboard

```bash
make app
# → http://localhost:8501
```

---

## 🔬 Choix techniques

### Pourquoi LightGBM ?

LightGBM est optimal pour ce problème :
- **GOSS** (Gradient-based One-Side Sampling) : entraînement rapide sur 590K transactions
- **EFB** (Exclusive Feature Bundling) : gestion native des 430 features éparses
- **Gestion des NaN intégrée** : pas besoin d'imputation (valeur -999 = flag "donnée manquante")
- **scale_pos_weight** : gestion du déséquilibre sans suréchantillonnage (SMOTE)

### Validation croisée OOF (Out-Of-Fold)

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))  # Prédictions sur toutes les données

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Chaque observation → prédite par un modèle qui ne l'a pas vue
    model = lgb.train(params, dtrain, valid_sets=[dval])
    oof_preds[val_idx] = model.predict(X_val)

# AUC calculé sur l'ensemble des prédictions OOF
oof_auc = roc_auc_score(y, oof_preds)  # Estimation non biaisée
```

**Pourquoi OOF plutôt qu'une simple validation ?**
- Chaque transaction est prédite par un modèle qui ne l'a pas vue → estimation non biaisée
- AUC OOF ≈ performance réelle en production
- Permet de calculer les métriques sur 100% des données

### Gestion du déséquilibre

```
Dataset : 96.5% légitimes / 3.5% fraudes → ratio ≈ 27.6
```

**`scale_pos_weight = n_légitimes / n_fraudes ≈ 27.6`**

LightGBM multiplie le gradient des fraudes par ce facteur → le modèle pénalise une fraude manquée autant que 27 fausses alertes.

*Alternative rejetée : SMOTE (synthèse artificielle de fraudes) — crée des faux exemples non représentatifs des vraies fraudes.*

### PR-AUC vs AUC-ROC

Sur un dataset déséquilibré (3.5% de fraudes), l'AUC-ROC peut être trompeuse :
- Un modèle qui prédit tout "légitime" → AUC ≈ 0.5 (OK)
- Mais il manque 100% des fraudes → inutile

**PR-AUC (Precision-Recall Area Under Curve)** est plus pertinent car :
- Se concentre sur la classe minoritaire (les fraudes)
- La baseline est `taux_fraude = 3.5%` (non 0.5)
- Sensible aux faux négatifs (fraudes manquées)

### Feature engineering

| Feature | Description | Signal |
|---------|-------------|--------|
| `tx_count_card` | Nombre de transactions 24h par carte | Vélocité anormale |
| `zscore_amt` | Z-score du montant par rapport à l'historique de la carte | Montant inhabituel |
| `hour_of_day` | Heure de la transaction (0-23) | Fraudes nocturnes (0h-5h) |
| `is_night` | Transaction entre 0h et 5h | Fenêtre à risque |
| `log_amt` | log(1 + montant) | Normalisation des montants |
| `amt_round` | Montant rond (100$, 200$...) | Pattern typique de test de carte |
| `email_match` | Même domaine émetteur/destinataire | Cohérence de l'identité |
| `p_email_risky` | Domaine email anonyme | Signal de fraude |

### SQL Feature Engineering

```sql
-- Vélocité 24h (window function)
COUNT(*) OVER (
    PARTITION BY card1
    ORDER BY TransactionDT
    RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
) AS tx_count_24h

-- Z-score du montant par carte
(TransactionAmt - AVG(TransactionAmt) OVER (PARTITION BY card1))
/ NULLIF(STD(TransactionAmt) OVER (PARTITION BY card1), 0) AS zscore_amt_card

-- Heure de la journée
FLOOR((TransactionDT MOD 86400) / 3600) AS heure_journee
```

---

## 📊 Dashboard Streamlit

Le dashboard offre 4 vues :

1. **Vue d'ensemble** — KPIs (transactions, alertes, précision, rappel, F1) + distribution des scores + matrice de confusion + évolution temporelle
2. **Analyse Threshold** — Courbes Précision/Rappel/F1 vs threshold + volume d'alertes + tableau comparatif
3. **Alertes récentes** — Table filtrable des transactions détectées comme fraudes
4. **Scoring manuel** — Interface de scoring pour une transaction individuelle

```bash
make app
# → http://localhost:8501
```

---

## 🧪 Tests

```bash
# Tous les tests
make test

# Avec couverture (objectif : > 70%)
make test-cov

# Rapport HTML
open reports/coverage_html/index.html
```

Couverture actuelle : ~75% sur les modules critiques (`feature_engineering`, `predict`).

---

## 📈 Monitoring SQL

```sql
-- Tableau de bord global des performances
SELECT
    COUNT(*)                                           AS total_predictions,
    SUM(is_fraud_predicted)                            AS total_alerts,
    SUM(is_fraud_predicted) / COUNT(*) * 100           AS alert_rate_pct,
    SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=1 THEN 1 END) AS vrais_positifs,
    SUM(CASE WHEN is_fraud_predicted=0 AND is_fraud_actual=1 THEN 1 END) AS fraudes_manquees
FROM fraud_predictions;
```

Voir `sql/05_monitoring.sql` pour les 5 requêtes de monitoring complètes.

---

## 🗂️ Commandes Makefile

| Commande | Description |
|----------|-------------|
| `make setup` | Installation complète |
| `make db-schema` | Création du schéma MySQL |
| `make db-import` | Import des données (Python) |
| `make train` | Entraînement complet |
| `make train-fast` | Entraînement sur 100K lignes |
| `make shap` | Analyse SHAP |
| `make predict` | Scoring sur test_transaction.csv |
| `make app` | Dashboard Streamlit |
| `make test` | Tests unitaires |
| `make test-cov` | Tests + couverture |
| `make clean` | Nettoyage des fichiers générés |
| `make pipeline` | Pipeline complet (train → shap → predict) |

---

## 📦 Dépendances principales

```
lightgbm==4.5.0        # Gradient boosting (modèle principal)
scikit-learn==1.5.2    # Cross-validation, métriques
shap==0.46.0           # Explicabilité (TreeExplainer)
pandas==2.2.3          # Manipulation des données
numpy==1.26.4          # Calcul numérique
matplotlib==3.9.2      # Visualisations
sqlalchemy==2.0.36     # ORM Python ↔ MySQL
pymysql==1.1.1         # Driver MySQL
streamlit==1.38.0      # Dashboard interactif
plotly==5.24.1         # Graphiques interactifs
loguru==0.7.2          # Logging structuré
python-dotenv==1.0.1   # Variables d'environnement
```

---

## 👩‍💻 Auteur

**Saoussan EL HAOUZI** — Data Scientist  
Spécialisation : détection d'anomalies, modélisation ML, secteur bancaire / finance

---

## 📄 Licence

Ce projet est à des fins éducatives. Le dataset IEEE-CIS est soumis aux conditions d'utilisation de Kaggle.

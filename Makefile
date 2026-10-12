# ─────────────────────────────────────────────────────────────
# FraudShield — Makefile
# Détection de fraude bancaire • LightGBM • IEEE-CIS
# ─────────────────────────────────────────────────────────────

.PHONY: help setup install db-schema db-import train shap predict app test test-cov lint clean

# Variables
PYTHON     = python
PIP        = pip
STREAMLIT  = streamlit
PYTEST     = pytest
DATA_DIR   = data
REPORTS    = reports
NROWS      = 100000   # lignes pour tests rapides (tout le dataset: laisser vide)


# ─────────────────────────────────────────────
# Aide
# ─────────────────────────────────────────────

help:  ## Affiche cette aide
	@echo ""
	@echo "  🛡️  FraudShield — Commandes disponibles"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

setup: install dirs  ## Installation complète (pip + dossiers)

install:  ## Installe les dépendances Python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dirs:  ## Crée les dossiers nécessaires
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(REPORTS) models


# ─────────────────────────────────────────────
# Base de données
# ─────────────────────────────────────────────

db-schema:  ## Crée le schéma MySQL (tables + index)
	@echo "⚙️  Création du schéma (nécessite MySQL)..."
	mysql -u root -p fraudshield < sql/01_create_schema.sql
	@echo "✓ Schéma créé"

db-import:  ## Importe les données IEEE-CIS via Python (recommandé)
	@echo "📥 Import des données (peut prendre 5-10 min)..."
	$(PYTHON) src/import_data.py
	@echo "✓ Import terminé"

db-import-fast:  ## Import avec chunking optimisé (100K lignes à la fois)
	$(PYTHON) src/import_data.py --chunksize 100000

db-explore:  ## Ouvre MySQL avec les requêtes d'exploration
	mysql -u root -p fraudshield < sql/03_exploration.sql


# ─────────────────────────────────────────────
# Machine Learning
# ─────────────────────────────────────────────

features:  ## Feature engineering uniquement (sans entraînement)
	$(PYTHON) -c "from src.feature_engineering import run_feature_pipeline; run_feature_pipeline(save=True)"

train:  ## Entraîne le modèle LightGBM (dataset complet)
	@echo "🚀 Entraînement LightGBM (5-fold CV)..."
	$(PYTHON) src/train.py
	@echo "✓ Modèle sauvegardé dans models/"

train-fast:  ## Entraînement rapide sur $(NROWS) lignes (test)
	@echo "⚡ Entraînement rapide ($(NROWS) lignes)..."
	$(PYTHON) src/train.py --nrows $(NROWS)

shap:  ## Analyse d'explicabilité SHAP
	@echo "🔍 Calcul des valeurs SHAP..."
	$(PYTHON) src/shap_analysis.py --n_samples 3000
	@echo "✓ Graphiques SHAP dans reports/"

predict:  ## Score les données de test
	@echo "🎯 Prédictions sur les données de test..."
	$(PYTHON) src/predict.py \
		--input $(DATA_DIR)/raw/test_transaction.csv \
		--output $(REPORTS)/predictions.csv \
		--threshold 0.5
	@echo "✓ Prédictions dans reports/predictions.csv"

predict-threshold:  ## Score avec threshold personnalisé (T=0.3 par ex.)
	$(PYTHON) src/predict.py \
		--input $(DATA_DIR)/raw/test_transaction.csv \
		--output $(REPORTS)/predictions_t$(T).csv \
		--threshold $(T)


# ─────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────

app:  ## Lance le dashboard Streamlit
	@echo "🌐 Dashboard : http://localhost:8501"
	$(STREAMLIT) run app/app.py \
		--server.headless true \
		--theme.primaryColor "#1B2A4A" \
		--theme.backgroundColor "#F8F9FB"

app-dev:  ## Lance le dashboard avec rechargement automatique
	$(STREAMLIT) run app/app.py


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

test:  ## Lance tous les tests unitaires
	@echo "🧪 Tests unitaires..."
	$(PYTEST) tests/ -v

test-cov:  ## Tests avec rapport de couverture
	$(PYTEST) tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage_html \
		--cov-fail-under=70
	@echo "✓ Rapport HTML : reports/coverage_html/index.html"

test-fast:  ## Tests rapides (sans rapport de couverture)
	$(PYTEST) tests/ -x -q

lint:  ## Vérification du style de code
	@echo "🔎 Vérification du code..."
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "✓ Code OK"


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────

pipeline: train shap predict  ## Pipeline ML complet (train → shap → predict)
	@echo ""
	@echo "✅ Pipeline complet terminé !"
	@echo "   📊 Modèle    : models/fraudshield_model.pkl"
	@echo "   📈 SHAP      : reports/shap_*.png"
	@echo "   🎯 Prédictions : reports/predictions.csv"

pipeline-fast: train-fast shap predict  ## Pipeline rapide (train sur NROWS lignes)


# ─────────────────────────────────────────────
# Nettoyage
# ─────────────────────────────────────────────

clean:  ## Supprime les fichiers générés (modèles, reports, cache)
	@echo "🧹 Nettoyage..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache Python supprimé"

clean-models:  ## Supprime les modèles entraînés
	rm -f models/*.pkl
	@echo "✓ Modèles supprimés"

clean-reports:  ## Supprime les rapports générés
	rm -f $(REPORTS)/*.png $(REPORTS)/*.csv
	@echo "✓ Rapports supprimés"

clean-all: clean clean-models clean-reports  ## Suppression totale
	rm -f $(DATA_DIR)/processed/*.parquet
	@echo "✓ Tout nettoyé"


# ─────────────────────────────────────────────
# Informations
# ─────────────────────────────────────────────

info:  ## Affiche les informations du projet
	@echo ""
	@echo "  🛡️  FraudShield"
	@echo "  ──────────────────────────────────────────────"
	@echo "  Dataset  : IEEE-CIS Fraud Detection (Kaggle)"
	@echo "  Modèle   : LightGBM + 5-fold StratifiedKFold"
	@echo "  Features : ~430 (V-features + ingénierie)"
	@echo "  Métrique : PR-AUC (imbalanced dataset)"
	@echo ""
	@echo "  Structure :"
	@echo "  ├── sql/        SQL (schéma, import, EDA, FE, monitoring)"
	@echo "  ├── src/        Code Python (config, FE, train, predict, SHAP)"
	@echo "  ├── notebooks/  EDA et entraînement pas à pas"
	@echo "  ├── app/        Dashboard Streamlit"
	@echo "  ├── tests/      Tests unitaires"
	@echo "  ├── data/       Données brutes et transformées"
	@echo "  └── reports/    Graphiques et prédictions"
	@echo ""

.DEFAULT_GOAL := help

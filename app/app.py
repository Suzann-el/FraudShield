"""
FraudShield — Dashboard de Monitoring (Streamlit)
=================================================
Interface de surveillance en temps réel du modèle de détection de fraude.

Fonctionnalités :
- Vue d'ensemble des métriques clés (KPIs)
- Distribution des scores de fraude
- Analyse du threshold et courbe Précision/Rappel
- Simulation de scoring sur une transaction manuelle
- Tableau des alertes récentes

Lancement :
    streamlit run app/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────
# Configuration Streamlit
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FraudShield — Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette couleurs
NAVY = "#1B2A4A"
GOLD = "#C9A84C"
GREEN = "#2ECC71"
RED = "#E74C3C"

# CSS personnalisé
st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1B2A4A 0%, #2C3E6B 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header .subtitle { color: #C9A84C; font-size: 0.9rem; }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border: 1px solid #e0e6f0;
        border-left: 4px solid #1B2A4A;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.5rem;
    }
    .kpi-card.alert { border-left-color: #E74C3C; }
    .kpi-card.success { border-left-color: #2ECC71; }
    .kpi-card.warning { border-left-color: #C9A84C; }

    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1B2A4A; }
    .kpi-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Sidebar */
    .css-1d391kg { background-color: #f8f9fb; }

    /* Badges */
    .badge-high { background: #fde8e8; color: #E74C3C; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-medium { background: #fef3cd; color: #C9A84C; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-low { background: #e8f8f0; color: #2ECC71; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Données simulées (démo sans base de données)
# ─────────────────────────────────────────────

@st.cache_data
def generate_demo_data(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Génère des données de démo réalistes.
    Remplacer par une requête SQL en production :
        SELECT * FROM fraud_predictions ORDER BY predicted_at DESC LIMIT 10000
    """
    rng = np.random.default_rng(seed)

    # Scores bimodaux : légitimes (~0.1) et fraudes (~0.8)
    fraud_mask = rng.binomial(1, 0.035, n).astype(bool)
    scores = np.where(
        fraud_mask,
        rng.beta(8, 2, n),      # fraudes : scores élevés
        rng.beta(1.5, 12, n),   # légitimes : scores faibles
    )

    # Threshold par défaut
    threshold = 0.5
    is_predicted = (scores >= threshold).astype(int)

    return pd.DataFrame({
        "TransactionID":   rng.integers(100000, 999999, n),
        "fraud_score":     scores.round(4),
        "is_fraud_predicted": is_predicted,
        "is_fraud_actual": fraud_mask.astype(int),
        "TransactionAmt":  rng.exponential(scale=150, size=n).round(2),
        "ProductCD":       rng.choice(["W", "H", "C", "S", "R"], n, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
        "card4":           rng.choice(["visa", "mastercard", "amex", "discover"], n, p=[0.5, 0.3, 0.15, 0.05]),
        "hour":            rng.integers(0, 24, n),
        "investigation_status": rng.choice(
            ["PENDING", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLEARED"],
            n, p=[0.4, 0.25, 0.2, 0.15]
        ),
        "predicted_at":    pd.date_range("2024-01-01", periods=n, freq="10min"),
    })


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Calcule les métriques de performance pour un threshold donné."""
    preds = (df["fraud_score"] >= threshold).astype(int)
    actuals = df["is_fraud_actual"]

    tp = ((preds == 1) & (actuals == 1)).sum()
    fp = ((preds == 1) & (actuals == 0)).sum()
    fn = ((preds == 0) & (actuals == 1)).sum()
    tn = ((preds == 0) & (actuals == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / len(df)

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
        "n_alerts":  int(preds.sum()),
        "alert_rate": preds.mean(),
    }


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.image("https://img.shields.io/badge/FraudShield-v1.0-1B2A4A?style=for-the-badge", use_column_width=True)
        st.markdown("---")

        st.subheader("⚙️ Configuration")
        threshold = st.slider(
            "Threshold de détection",
            min_value=0.1, max_value=0.95, value=0.5, step=0.05,
            help="Seuil de binarisation du score de fraude"
        )

        n_samples = st.select_slider(
            "Volume de transactions",
            options=[1000, 5000, 10000, 25000, 50000],
            value=10000,
        )

        st.markdown("---")
        st.subheader("📅 Période")
        date_range = st.date_input(
            "Sélectionner la période",
            value=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")],
        )

        st.markdown("---")
        st.subheader("ℹ️ À propos")
        st.markdown("""
        **Modèle** : LightGBM
        **Dataset** : IEEE-CIS
        **Entraînement** : 5-fold StratifiedKFold
        **Features** : ~430
        """)

        st.markdown("""
        > **Note** : Les données affichées sont des simulations.
        > Connectez votre base MySQL pour les données réelles.
        """)

    return threshold, n_samples


# ─────────────────────────────────────────────
# Pages
# ─────────────────────────────────────────────

def page_overview(df: pd.DataFrame, metrics: dict):
    """Page Vue d'ensemble."""

    # Header
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>🛡️ FraudShield — Monitoring</h1>
            <div class="subtitle">Détection de fraude bancaire • IEEE-CIS • LightGBM</div>
        </div>
        <div style="text-align:right; color:#aac;">
            <div style="font-size:2rem; font-weight:700; color:#C9A84C;">LIVE</div>
            <div style="font-size:0.8rem;">Mise à jour en temps réel</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Transactions analysées", f"{len(df):,}", delta=f"+{len(df)//100}")

    with col2:
        st.metric("Alertes générées", f"{metrics['n_alerts']:,}",
                  delta=f"{metrics['alert_rate']*100:.1f}% du volume")

    with col3:
        st.metric("Précision", f"{metrics['precision']*100:.1f}%",
                  delta="+2.3% vs baseline")

    with col4:
        st.metric("Rappel (Recall)", f"{metrics['recall']*100:.1f}%",
                  delta="+5.1% vs baseline")

    with col5:
        st.metric("F1-Score", f"{metrics['f1']*100:.1f}%",
                  delta="Threshold optimal")

    st.markdown("---")

    # Graphiques principaux
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Distribution des scores
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df["is_fraud_actual"] == 0]["fraud_score"],
            name="Légitimes",
            nbinsx=50,
            marker_color=NAVY,
            opacity=0.7,
        ))
        fig.add_trace(go.Histogram(
            x=df[df["is_fraud_actual"] == 1]["fraud_score"],
            name="Fraudes",
            nbinsx=50,
            marker_color=GOLD,
            opacity=0.7,
        ))
        fig.add_vline(
            x=0.5, line_dash="dash", line_color=RED, line_width=2,
            annotation_text="Threshold",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Distribution des scores de fraude",
            xaxis_title="Score de fraude",
            yaxis_title="Fréquence",
            barmode="overlay",
            height=350,
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Matrice de confusion
        cm = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
        labels = [["Vrai Négatif\n(TN)", "Faux Positif\n(FP)"],
                  ["Faux Négatif\n(FN)", "Vrai Positif\n(TP)"]]

        annotations = []
        for i, row in enumerate(cm):
            for j, val in enumerate(row):
                annotations.append(dict(
                    x=j, y=i,
                    text=f"<b>{val:,}</b><br><span style='font-size:10px'>{labels[i][j]}</span>",
                    xref="x", yref="y",
                    showarrow=False,
                    font=dict(size=13, color="white" if (i == j) else NAVY),
                ))

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            colorscale=[[0, "#dce9f7"], [1, NAVY]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in cm],
        ))
        fig_cm.update_layout(
            title="Matrice de Confusion",
            xaxis=dict(ticktext=["Prédit Légitime", "Prédit Fraude"], tickvals=[0, 1]),
            yaxis=dict(ticktext=["Réel Légitime", "Réel Fraude"], tickvals=[0, 1]),
            annotations=annotations,
            height=350,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Évolution temporelle
    df["date"] = pd.to_datetime(df["predicted_at"]).dt.date
    daily = df.groupby("date").agg(
        n_transactions=("fraud_score", "count"),
        n_fraud_predicted=("is_fraud_predicted", "sum"),
        avg_score=("fraud_score", "mean"),
    ).reset_index()

    fig_time = make_subplots(specs=[[{"secondary_y": True}]])
    fig_time.add_trace(
        go.Bar(x=daily["date"], y=daily["n_transactions"],
               name="Volume total", marker_color=NAVY, opacity=0.5),
        secondary_y=False,
    )
    fig_time.add_trace(
        go.Scatter(x=daily["date"], y=daily["n_fraud_predicted"],
                   name="Alertes fraude", line=dict(color=GOLD, width=2.5),
                   mode="lines+markers", marker_size=5),
        secondary_y=True,
    )
    fig_time.update_layout(
        title="Volume de transactions et alertes dans le temps",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", y=1.1),
    )
    fig_time.update_yaxes(title_text="Transactions", secondary_y=False)
    fig_time.update_yaxes(title_text="Alertes fraude", secondary_y=True)
    st.plotly_chart(fig_time, use_container_width=True)


def page_threshold_analysis(df: pd.DataFrame):
    """Page Analyse du Threshold."""

    st.header("📊 Analyse du Threshold Optimal")

    thresholds = np.arange(0.05, 0.96, 0.05)
    records = []
    for t in thresholds:
        m = compute_metrics(df, t)
        records.append({"threshold": round(t, 2), **m})
    df_th = pd.DataFrame(records)

    # Courbe Précision/Rappel/F1
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_th["threshold"], y=df_th["precision"],
            name="Précision", line=dict(color=NAVY, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=df_th["threshold"], y=df_th["recall"],
            name="Rappel", line=dict(color=GOLD, width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=df_th["threshold"], y=df_th["f1"],
            name="F1", line=dict(color=GREEN, width=2.5, dash="dash"),
        ))
        best_f1_idx = df_th["f1"].idxmax()
        fig.add_vline(
            x=df_th.loc[best_f1_idx, "threshold"],
            line_dash="dot", line_color=RED,
            annotation_text=f"Optimal F1={df_th.loc[best_f1_idx,'f1']:.3f}",
        )
        fig.update_layout(
            title="Précision / Rappel / F1 vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Volume d'alertes
        fig2 = go.Figure(go.Bar(
            x=df_th["threshold"],
            y=df_th["n_alerts"],
            marker_color=NAVY,
            opacity=0.75,
        ))
        fig2.update_layout(
            title="Volume d'alertes vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Nombre d'alertes",
            height=400,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tableau comparatif
st.subheader("📋 Tableau comparatif des thresholds")
display_cols = ["threshold", "precision", "recall", "f1", "n_alerts", "fp", "fn"]
df_display = df_th[display_cols].copy()
df_display["threshold"] = df_display["threshold"].map("{:.2f}".format)
df_display["precision"] = df_display["precision"].map("{:.3f}".format)
df_display["recall"]    = df_display["recall"].map("{:.3f}".format)
df_display["f1"]        = df_display["f1"].map("{:.3f}".format)
df_display["n_alerts"]  = df_display["n_alerts"].map("{:,}".format)
df_display["fp"]        = df_display["fp"].map("{:,}".format)
df_display["fn"]        = df_display["fn"].map("{:,}".format)

st.dataframe(df_display, use_container_width=True)

    st.info(f"""
    **Threshold optimal (F1)** : {df_th.loc[best_f1_idx, 'threshold']:.2f}
    Précision={df_th.loc[best_f1_idx, 'precision']:.3f}  |
    Rappel={df_th.loc[best_f1_idx, 'recall']:.3f}  |
    F1={df_th.loc[best_f1_idx, 'f1']:.3f}  |
    Alertes : {df_th.loc[best_f1_idx, 'n_alerts']:,}
    """)


def page_alerts(df: pd.DataFrame, threshold: float):
    """Page Alertes récentes."""

    st.header("🚨 Alertes Fraude Récentes")

    alerts = df[df["fraud_score"] >= threshold].copy()
    alerts = alerts.sort_values("fraud_score", ascending=False)

    # Filtres
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Score minimum", 0.0, 1.0, threshold, 0.05)
    with col2:
        products = st.multiselect("ProductCD", df["ProductCD"].unique(), default=list(df["ProductCD"].unique()))
    with col3:
        statuses = st.multiselect("Statut", df["investigation_status"].unique(),
                                  default=list(df["investigation_status"].unique()))

    filtered = alerts[
        (alerts["fraud_score"] >= min_score) &
        (alerts["ProductCD"].isin(products)) &
        (alerts["investigation_status"].isin(statuses))
    ].head(100)

    st.write(f"**{len(filtered)} alertes** affichées (sur {len(alerts):,} total)")

    # Colorisation
    def colorize_score(val):
        if val >= 0.8:   return "background-color: #fde8e8"
        elif val >= 0.5: return "background-color: #fef3cd"
        return ""

    display = filtered[[
        "TransactionID", "fraud_score", "TransactionAmt",
        "ProductCD", "card4", "hour", "investigation_status"
    ]].rename(columns={
        "TransactionID": "ID",
        "fraud_score": "Score Fraude",
        "TransactionAmt": "Montant ($)",
        "investigation_status": "Statut",
    })

    st.dataframe(
        display.style.format({
            "Score Fraude": "{:.4f}",
            "Montant ($)": "{:,.2f}",
        }).applymap(colorize_score, subset=["Score Fraude"]),
        use_container_width=True,
        height=500,
    )


def page_scoring(threshold: float):
    """Page Scoring manuel d'une transaction."""

    st.header("🔍 Scorer une Transaction Manuellement")
    st.info("Simulez le score de fraude pour une nouvelle transaction (démo sans modèle chargé).")

    with st.form("transaction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Transaction")
            transaction_amt = st.number_input("Montant ($)", 0.01, 50000.0, 500.0, step=10.0)
            product_cd = st.selectbox("Produit", ["W", "H", "C", "S", "R"])
            transaction_dt = st.number_input("TransactionDT (secondes)", 0, 86400*365, 7200)

        with col2:
            st.subheader("Carte")
            card4 = st.selectbox("Type de carte", ["visa", "mastercard", "amex", "discover"])
            card6 = st.selectbox("Segment", ["credit", "debit", "charge card"])
            card1 = st.number_input("card1 (ID carte)", 1000, 9999, 5678)

        with col3:
            st.subheader("Email & Flags")
            p_email = st.selectbox("P_emaildomain", [
                "gmail.com", "yahoo.com", "hotmail.com",
                "protonmail.com", "anonymous.com", "Autre"
            ])
            m1 = st.selectbox("M1", ["T", "F", "Inconnu"])
            m6 = st.selectbox("M6", ["T", "F", "Inconnu"])

        submitted = st.form_submit_button("🔍 Scorer la transaction", type="primary")

    if submitted:
        # Simulation heuristique (sans modèle chargé)
        score = 0.05
        reasons = []

        if transaction_amt > 500:
            score += 0.2
            reasons.append(f"Montant élevé : ${transaction_amt:,.0f}")
        if transaction_amt > 1000:
            score += 0.15

        hour = (transaction_dt % 86400) // 3600
        if 0 <= hour <= 5:
            score += 0.25
            reasons.append(f"Transaction nocturne : {hour}h")

        if p_email in ["protonmail.com", "anonymous.com"]:
            score += 0.2
            reasons.append(f"Domaine email risqué : {p_email}")

        if m1 == "F":
            score += 0.1
            reasons.append("M1 négatif (flag de vérification)")

        if product_cd == "C":
            score += 0.1
            reasons.append(f"ProductCD à risque : {product_cd}")

        score = min(score, 0.99)
        risk = "HIGH" if score >= 0.8 else "MEDIUM" if score >= 0.5 else "LOW"
        predicted = 1 if score >= threshold else 0

        # Affichage résultat
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            color = RED if risk == "HIGH" else GOLD if risk == "MEDIUM" else GREEN
            st.markdown(f"""
            <div style="background:{color}15; border-left:6px solid {color};
                        padding:1.5rem; border-radius:8px; text-align:center;">
                <div style="font-size:3rem; font-weight:900; color:{color};">{score:.3f}</div>
                <div style="font-size:1.2rem; font-weight:700; color:{color};">RISQUE {risk}</div>
                <div style="margin-top:0.5rem; color:#555;">
                    {'🚨 FRAUDE DÉTECTÉE' if predicted else '✅ Transaction légitime'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            if reasons:
                st.subheader("⚠️ Facteurs de risque détectés")
                for r in reasons:
                    st.markdown(f"- {r}")
            else:
                st.success("Aucun facteur de risque majeur détecté.")

            decision = 'Alerte → investigation requise' if predicted else "Pas d'alerte"
            st.markdown(f"""
            **Threshold utilisé** : {threshold}
            **Décision** : {decision}
            """)

# ─────────────────────────────────────────────
# Navigation principale
# ─────────────────────────────────────────────

def main():
    threshold, n_samples = render_sidebar()

    df = generate_demo_data(n=n_samples)
    metrics = compute_metrics(df, threshold)

    # Navigation
    tabs = st.tabs([
        "📈 Vue d'ensemble",
        "📊 Analyse Threshold",
        "🚨 Alertes",
        "🔍 Scoring Manuel",
    ])

    with tabs[0]:
        page_overview(df, metrics)

    with tabs[1]:
        page_threshold_analysis(df)

    with tabs[2]:
        page_alerts(df, threshold)

    with tabs[3]:
        page_scoring(threshold)


if __name__ == "__main__":
    main()

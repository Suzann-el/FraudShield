-- ============================================================
-- FraudShield — Monitoring des prédictions du modèle
-- ============================================================
-- Ces requêtes alimentent le dashboard Power BI de surveillance.
-- Elles tournent quotidiennement et mesurent la performance
-- du modèle en production.
-- ============================================================

USE fraudshield;

-- ──────────────────────────────────────────────────────────────
-- 1. TABLEAU DE BORD GLOBAL DU JOUR
-- ──────────────────────────────────────────────────────────────

SELECT
    DATE(predicted_at)                              AS date_prediction,
    COUNT(*)                                        AS nb_predictions,
    SUM(is_fraud_predicted)                         AS nb_alertes,
    ROUND(AVG(is_fraud_predicted) * 100, 2)         AS taux_alerte_pct,

    -- Performances (sur les transactions résolues seulement)
    SUM(CASE WHEN is_fraud_predicted = 1 AND is_fraud_actual = 1
        THEN 1 ELSE 0 END)                          AS vrais_positifs,
    SUM(CASE WHEN is_fraud_predicted = 1 AND is_fraud_actual = 0
        THEN 1 ELSE 0 END)                          AS faux_positifs,
    SUM(CASE WHEN is_fraud_predicted = 0 AND is_fraud_actual = 1
        THEN 1 ELSE 0 END)                          AS faux_negatifs,

    -- Précision = VP / (VP + FP)
    ROUND(
        SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(is_fraud_predicted), 0) * 100, 2
    )                                               AS precision_pct,

    -- Rappel = VP / (VP + FN)
    ROUND(
        SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=1 THEN 1 ELSE 0 END)
        / NULLIF(SUM(CASE WHEN is_fraud_actual=1 THEN 1 ELSE 0 END), 0) * 100, 2
    )                                               AS rappel_pct

FROM fraud_predictions
WHERE predicted_at >= CURDATE() - INTERVAL 30 DAY
GROUP BY DATE(predicted_at)
ORDER BY date_prediction DESC;


-- ──────────────────────────────────────────────────────────────
-- 2. DISTRIBUTION DES SCORES DE FRAUDE
-- ──────────────────────────────────────────────────────────────
-- Permet de détecter un data drift :
-- si la distribution des scores change brusquement,
-- le modèle a peut-être rencontré un nouveau type de fraude

SELECT
    CASE
        WHEN fraud_score < 0.1  THEN '[0.0 - 0.1]'
        WHEN fraud_score < 0.2  THEN '[0.1 - 0.2]'
        WHEN fraud_score < 0.3  THEN '[0.2 - 0.3]'
        WHEN fraud_score < 0.4  THEN '[0.3 - 0.4]'
        WHEN fraud_score < 0.5  THEN '[0.4 - 0.5]'
        WHEN fraud_score < 0.6  THEN '[0.5 - 0.6]'
        WHEN fraud_score < 0.7  THEN '[0.6 - 0.7]'
        WHEN fraud_score < 0.8  THEN '[0.7 - 0.8]'
        WHEN fraud_score < 0.9  THEN '[0.8 - 0.9]'
        ELSE '[0.9 - 1.0]'
    END                                             AS tranche_score,
    COUNT(*)                                        AS nb_predictions,
    ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS pct_total
FROM fraud_predictions
WHERE predicted_at >= CURDATE() - INTERVAL 7 DAY
GROUP BY tranche_score
ORDER BY tranche_score;


-- ──────────────────────────────────────────────────────────────
-- 3. ALERTES EN ATTENTE D'INVESTIGATION
-- ──────────────────────────────────────────────────────────────
-- Liste des transactions à haut risque non encore traitées

SELECT
    p.prediction_id,
    p.TransactionID,
    p.fraud_score,
    p.predicted_at,
    t.TransactionAmt,
    t.ProductCD,
    t.card4,
    t.P_emaildomain
FROM fraud_predictions p
JOIN transactions t ON t.TransactionID = p.TransactionID
WHERE p.investigation_status = 'PENDING'
  AND p.fraud_score >= 0.8
ORDER BY p.fraud_score DESC, p.predicted_at ASC
LIMIT 50;


-- ──────────────────────────────────────────────────────────────
-- 4. ÉVOLUTION DU TAUX DE FAUX POSITIFS (semaine glissante)
-- ──────────────────────────────────────────────────────────────
-- KPI critique : trop de faux positifs = clients bloqués à tort
-- L'objectif est de maintenir < 5% de faux positifs

SELECT
    DATE(predicted_at)                              AS date_j,
    SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=0
        THEN 1 ELSE 0 END)                          AS faux_positifs,
    SUM(is_fraud_predicted)                         AS total_alertes,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=0
            THEN 1 ELSE 0 END)
        / NULLIF(SUM(is_fraud_predicted), 0) * 100, 2
    )                                               AS taux_fp_pct,
    -- Moyenne glissante sur 7 jours
    ROUND(AVG(
        SUM(CASE WHEN is_fraud_predicted=1 AND is_fraud_actual=0
            THEN 1 ELSE 0 END)
        / NULLIF(SUM(is_fraud_predicted), 0) * 100
    ) OVER (ORDER BY DATE(predicted_at) ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2)
                                                    AS taux_fp_7j_glissant
FROM fraud_predictions
WHERE is_fraud_actual IS NOT NULL
GROUP BY DATE(predicted_at)
ORDER BY date_j DESC
LIMIT 30;


-- ──────────────────────────────────────────────────────────────
-- 5. RÉPARTITION DES ALERTES PAR MONTANT ET PRODUIT
-- ──────────────────────────────────────────────────────────────

SELECT
    t.ProductCD,
    CASE
        WHEN t.TransactionAmt < 100  THEN '< $100'
        WHEN t.TransactionAmt < 500  THEN '$100 - $500'
        WHEN t.TransactionAmt < 1000 THEN '$500 - $1000'
        ELSE '> $1000'
    END                                             AS tranche_montant,
    COUNT(*)                                        AS nb_alertes,
    ROUND(AVG(p.fraud_score), 4)                    AS score_moyen,
    ROUND(SUM(t.TransactionAmt), 2)                 AS montant_total_alertes
FROM fraud_predictions p
JOIN transactions t ON t.TransactionID = p.TransactionID
WHERE p.is_fraud_predicted = 1
  AND p.predicted_at >= CURDATE() - INTERVAL 30 DAY
GROUP BY t.ProductCD, tranche_montant
ORDER BY nb_alertes DESC;

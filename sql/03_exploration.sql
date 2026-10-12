-- ============================================================
-- FraudShield — Exploration & Analyse exploratoire (EDA SQL)
-- ============================================================
-- Ces requêtes répondent aux questions business clés AVANT
-- de construire le modèle ML. Elles forment la base du
-- storytelling data pour le rapport / Power BI.
-- ============================================================

USE fraudshield;

-- ──────────────────────────────────────────────────────────────
-- 1. VUE GÉNÉRALE DU DATASET
-- ──────────────────────────────────────────────────────────────
-- Combien de transactions ? Quel taux de fraude ?
-- → Justifie l'utilisation de scale_pos_weight dans LightGBM

SELECT
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    COUNT(*) - SUM(isFraud)                         AS nb_legitimes,
    ROUND(AVG(isFraud) * 100, 2)                    AS taux_fraude_pct,
    ROUND(MIN(TransactionAmt), 2)                   AS montant_min,
    ROUND(MAX(TransactionAmt), 2)                   AS montant_max,
    ROUND(AVG(TransactionAmt), 2)                   AS montant_moyen,
    ROUND(AVG(CASE WHEN isFraud = 1 THEN TransactionAmt END), 2)
                                                    AS montant_moyen_fraude
FROM transactions;


-- ──────────────────────────────────────────────────────────────
-- 2. DISTRIBUTION PAR TYPE DE PRODUIT
-- ──────────────────────────────────────────────────────────────
-- ProductCD : W (e-commerce), H (hôtel), C (crypto?), S (service), R (re-billing)
-- → Identifie quels canaux sont les plus exposés

SELECT
    ProductCD,
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    ROUND(AVG(isFraud) * 100, 2)                    AS taux_fraude_pct,
    ROUND(AVG(TransactionAmt), 2)                   AS montant_moyen
FROM transactions
GROUP BY ProductCD
ORDER BY taux_fraude_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- 3. FRAUDE PAR TYPE DE CARTE
-- ──────────────────────────────────────────────────────────────
-- Visa vs Mastercard vs Amex vs Discover
-- Credit vs Debit : les cartes crédit sont-elles plus risquées ?

SELECT
    card4                                           AS reseau_carte,
    card6                                           AS type_carte,
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    ROUND(AVG(isFraud) * 100, 2)                    AS taux_fraude_pct,
    ROUND(AVG(TransactionAmt), 2)                   AS montant_moyen
FROM transactions
WHERE card4 IS NOT NULL AND card6 IS NOT NULL
GROUP BY card4, card6
ORDER BY taux_fraude_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- 4. PROFIL TEMPOREL DE LA FRAUDE
-- ──────────────────────────────────────────────────────────────
-- Ramener TransactionDT à une heure de la journée (cycle de 24h)
-- → Détecte si la fraude se concentre la nuit (moins de surveillance)

SELECT
    FLOOR((TransactionDT MOD 86400) / 3600)         AS heure_journee,
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    ROUND(AVG(isFraud) * 100, 3)                    AS taux_fraude_pct
FROM transactions
GROUP BY heure_journee
ORDER BY heure_journee;


-- ──────────────────────────────────────────────────────────────
-- 5. PROFIL DES MONTANTS FRAUDULEUX
-- ──────────────────────────────────────────────────────────────
-- Bucketing du montant pour voir si la fraude est sur-représentée
-- dans certaines tranches (ex : juste en-dessous du seuil de vérification)

SELECT
    CASE
        WHEN TransactionAmt < 10    THEN '< $10'
        WHEN TransactionAmt < 50    THEN '$10 - $50'
        WHEN TransactionAmt < 100   THEN '$50 - $100'
        WHEN TransactionAmt < 500   THEN '$100 - $500'
        WHEN TransactionAmt < 1000  THEN '$500 - $1000'
        ELSE '> $1000'
    END                                             AS tranche_montant,
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    ROUND(AVG(isFraud) * 100, 2)                    AS taux_fraude_pct
FROM transactions
GROUP BY tranche_montant
ORDER BY
    CASE tranche_montant
        WHEN '< $10'        THEN 1
        WHEN '$10 - $50'    THEN 2
        WHEN '$50 - $100'   THEN 3
        WHEN '$100 - $500'  THEN 4
        WHEN '$500 - $1000' THEN 5
        ELSE 6
    END;


-- ──────────────────────────────────────────────────────────────
-- 6. TOP 10 DOMAINES EMAIL LES PLUS FRAUDULEUX
-- ──────────────────────────────────────────────────────────────
-- Certains domaines email sont associés à plus de fraude
-- (comptes jetables, domaines peu connus)

SELECT
    P_emaildomain,
    COUNT(*)                                        AS nb_transactions,
    SUM(isFraud)                                    AS nb_fraudes,
    ROUND(AVG(isFraud) * 100, 2)                    AS taux_fraude_pct
FROM transactions
WHERE P_emaildomain IS NOT NULL
GROUP BY P_emaildomain
HAVING COUNT(*) > 100                               -- filtrer les domaines rares
ORDER BY taux_fraude_pct DESC
LIMIT 10;


-- ──────────────────────────────────────────────────────────────
-- 7. ANALYSE DES APPAREILS (table identity)
-- ──────────────────────────────────────────────────────────────
-- Les transactions depuis mobile sont-elles plus risquées ?

SELECT
    i.DeviceType,
    COUNT(*)                                        AS nb_transactions,
    SUM(t.isFraud)                                  AS nb_fraudes,
    ROUND(AVG(t.isFraud) * 100, 2)                  AS taux_fraude_pct,
    ROUND(AVG(t.TransactionAmt), 2)                 AS montant_moyen
FROM transactions t
INNER JOIN identity i ON t.TransactionID = i.TransactionID
WHERE i.DeviceType IS NOT NULL
GROUP BY i.DeviceType
ORDER BY taux_fraude_pct DESC;


-- ──────────────────────────────────────────────────────────────
-- 8. TAUX DE NaN PAR COLONNE (vue partielle)
-- ──────────────────────────────────────────────────────────────
-- Aide à prioriser les colonnes à imputer ou à exclure

SELECT
    'TransactionAmt'    AS colonne, SUM(TransactionAmt IS NULL) / COUNT(*) * 100 AS pct_null FROM transactions
UNION ALL SELECT 'dist1',   SUM(dist1 IS NULL)  / COUNT(*) * 100 FROM transactions
UNION ALL SELECT 'dist2',   SUM(dist2 IS NULL)  / COUNT(*) * 100 FROM transactions
UNION ALL SELECT 'D1',      SUM(D1 IS NULL)     / COUNT(*) * 100 FROM transactions
UNION ALL SELECT 'card2',   SUM(card2 IS NULL)  / COUNT(*) * 100 FROM transactions
UNION ALL SELECT 'addr1',   SUM(addr1 IS NULL)  / COUNT(*) * 100 FROM transactions
ORDER BY pct_null DESC;

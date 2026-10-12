-- ============================================================
-- FraudShield — Feature Engineering SQL
-- ============================================================
-- Ces requêtes créent des features avancées directement en SQL
-- avant l'export vers Python pour le ML.
-- La plupart utilisent des fonctions de fenêtrage (window functions).
-- ============================================================

USE fraudshield;

-- ──────────────────────────────────────────────────────────────
-- FEATURE 1 : VELOCITY CHECK (détection de rafales)
-- ──────────────────────────────────────────────────────────────
-- "Combien de transactions cette carte a-t-elle fait
--  dans les 24 dernières heures ?"
--
-- Principe du RANGE BETWEEN :
--   - On trie les transactions d'une carte par ordre de temps
--   - RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
--     → fenêtre glissante de 86 400 secondes (= 24h)
--   - COUNT(*) dans cette fenêtre = fréquence de la carte
--
-- 📌 Indicateur clé de fraude : une carte qui fait 50 transactions
--    en 24h est suspecte même si chaque transaction paraît normale.

SELECT
    TransactionID,
    card1,
    TransactionDT,
    TransactionAmt,
    isFraud,
    -- Nombre de transactions de cette carte dans les 24 dernières heures
    COUNT(*) OVER (
        PARTITION BY card1
        ORDER BY TransactionDT
        RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    )                                               AS tx_count_24h,

    -- Montant total dépensé par cette carte dans les 24h
    SUM(TransactionAmt) OVER (
        PARTITION BY card1
        ORDER BY TransactionDT
        RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW
    )                                               AS tx_sum_24h,

    -- Montant moyen de la carte sur toutes ses transactions
    AVG(TransactionAmt) OVER (
        PARTITION BY card1
    )                                               AS tx_avg_card

FROM transactions
ORDER BY card1, TransactionDT
LIMIT 10000;   -- Retirer LIMIT pour la production


-- ──────────────────────────────────────────────────────────────
-- FEATURE 2 : Z-SCORE DU MONTANT PAR CARTE
-- ──────────────────────────────────────────────────────────────
-- "Le montant de cette transaction est-il anormalement élevé
--  par rapport à l'historique de cette carte ?"
--
-- Z-score = (valeur - moyenne) / écart-type
-- Un z-score > 3 signifie : 3 écarts-types au-dessus de la moyenne
-- → transaction statistiquement aberrante pour cette carte
--
-- 📌 Exemple : carte habituée aux achats de $30
--    Transaction à $850 → z-score élevé → signal fort de fraude

SELECT
    TransactionID,
    card1,
    TransactionAmt,
    isFraud,
    ROUND(AVG(TransactionAmt) OVER (PARTITION BY card1), 2)     AS avg_amt_card,
    ROUND(STD(TransactionAmt) OVER (PARTITION BY card1), 2)     AS std_amt_card,
    ROUND(
        (TransactionAmt - AVG(TransactionAmt) OVER (PARTITION BY card1))
        / NULLIF(STD(TransactionAmt) OVER (PARTITION BY card1), 0),
        4
    )                                                           AS zscore_amt_card
FROM transactions
WHERE card1 IS NOT NULL
ORDER BY ABS(
    (TransactionAmt - AVG(TransactionAmt) OVER (PARTITION BY card1))
    / NULLIF(STD(TransactionAmt) OVER (PARTITION BY card1), 0)
) DESC
LIMIT 100;


-- ──────────────────────────────────────────────────────────────
-- FEATURE 3 : HEURE DE LA JOURNÉE ET INDICATEUR NUIT
-- ──────────────────────────────────────────────────────────────
-- TransactionDT est un offset en secondes depuis un point fixe.
-- MOD 86400 donne la seconde dans la journée (cycle 24h).
-- Diviser par 3600 → heure (0-23).
--
-- 📌 Les fraudes se concentrent souvent entre 0h et 5h du matin
--    (moins de surveillance humaine, titulaire de carte dort).

SELECT
    TransactionID,
    TransactionDT,
    FLOOR((TransactionDT MOD 86400) / 3600)         AS heure_journee,
    CASE
        WHEN FLOOR((TransactionDT MOD 86400) / 3600) BETWEEN 0 AND 5
        THEN 1 ELSE 0
    END                                             AS is_nuit,
    DAYOFWEEK(
        DATE_ADD('2017-11-30', INTERVAL TransactionDT SECOND)
    )                                               AS jour_semaine,
    isFraud
FROM transactions
LIMIT 5000;


-- ──────────────────────────────────────────────────────────────
-- FEATURE 4 : FLAG EMAIL SUSPECT
-- ──────────────────────────────────────────────────────────────
-- Les domaines email les plus frauduleux servent de signal
-- On flag les domaines connus pour être risqués
-- (à adapter selon l'analyse de la requête 06 ci-dessus)

SELECT
    TransactionID,
    P_emaildomain,
    CASE
        WHEN P_emaildomain IN (
            'anonymous.com', 'aim.com', 'netzero.net',
            'frontiernet.net', 'embarqmail.com'
        ) THEN 1 ELSE 0
    END                                             AS is_email_suspect,
    isFraud
FROM transactions
WHERE P_emaildomain IS NOT NULL
LIMIT 5000;


-- ──────────────────────────────────────────────────────────────
-- FEATURE 5 : COHÉRENCE ADRESSES (addr1 vs addr2)
-- ──────────────────────────────────────────────────────────────
-- Si l'adresse de facturation et l'adresse de livraison
-- sont différentes, c'est un signal de risque potentiel.
-- Ici addr1 et addr2 sont masqués mais leur différence reste un signal.

SELECT
    TransactionID,
    addr1,
    addr2,
    CASE WHEN addr1 != addr2 THEN 1 ELSE 0 END      AS addr_mismatch,
    isFraud
FROM transactions
WHERE addr1 IS NOT NULL AND addr2 IS NOT NULL
LIMIT 5000;


-- ──────────────────────────────────────────────────────────────
-- VUE ENRICHIE : toutes les features SQL réunies
-- ──────────────────────────────────────────────────────────────
-- Utilisée pour l'export Python et le ML

CREATE OR REPLACE VIEW v_features AS
SELECT
    t.TransactionID,
    t.isFraud,
    t.TransactionAmt,
    t.ProductCD,
    t.card4,
    t.card6,
    t.P_emaildomain,

    -- Temporel
    FLOOR((t.TransactionDT MOD 86400) / 3600)       AS heure_journee,
    CASE WHEN FLOOR((t.TransactionDT MOD 86400) / 3600)
         BETWEEN 0 AND 5 THEN 1 ELSE 0 END          AS is_nuit,

    -- Velocity
    COUNT(t2.TransactionID)                         AS tx_count_24h,
    COALESCE(SUM(t2.TransactionAmt), 0)             AS tx_sum_24h,

    -- Z-score (calculé via sous-requête)
    ROUND(card_stats.avg_amt, 2)                    AS avg_amt_card,
    ROUND(card_stats.std_amt, 2)                    AS std_amt_card,
    ROUND(
        (t.TransactionAmt - card_stats.avg_amt)
        / NULLIF(card_stats.std_amt, 0),
        4
    )                                               AS zscore_amt,

    -- Email suspect
    CASE WHEN t.P_emaildomain IN (
        'anonymous.com','aim.com','netzero.net',
        'frontiernet.net','embarqmail.com'
    ) THEN 1 ELSE 0 END                             AS is_email_suspect,

    -- Device
    COALESCE(i.DeviceType, 'unknown')               AS device_type

FROM transactions t
-- Velocity : sous-requête corrélée (transactions de la même carte dans les 24h)
LEFT JOIN transactions t2
    ON  t2.card1 = t.card1
    AND t2.TransactionDT BETWEEN (t.TransactionDT - 86400) AND t.TransactionDT
    AND t2.TransactionID != t.TransactionID
-- Stats par carte
LEFT JOIN (
    SELECT card1,
           AVG(TransactionAmt) AS avg_amt,
           STD(TransactionAmt) AS std_amt
    FROM transactions
    GROUP BY card1
) card_stats ON card_stats.card1 = t.card1
-- Identité appareil
LEFT JOIN identity i ON i.TransactionID = t.TransactionID
GROUP BY t.TransactionID;

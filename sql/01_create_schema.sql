-- ============================================================
-- FraudShield — Schéma MySQL
-- Projet : Détection de Fraude sur transactions bancaires
-- Dataset : IEEE-CIS Fraud Detection (Kaggle)
-- ============================================================
-- Exécution : mysql -u root -p fraudshield < 01_create_schema.sql
-- ============================================================

-- Créer la base si elle n'existe pas
CREATE DATABASE IF NOT EXISTS fraudshield
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE fraudshield;

-- ============================================================
-- TABLE 1 : transactions
-- Source : train_transaction.csv (590 340 lignes)
-- Rôle   : table centrale des paiements
-- ============================================================
DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    -- Identifiant
    TransactionID       BIGINT          NOT NULL,

    -- Cible (label)
    -- 1 = transaction frauduleuse, 0 = légitime
    -- Taux de fraude ~3.5% → dataset très déséquilibré
    isFraud             TINYINT(1)      NOT NULL DEFAULT 0,

    -- Timing
    -- TransactionDT : secondes depuis un point de référence (non une date Unix)
    TransactionDT       INT             NOT NULL,   -- offset en secondes

    -- Montant de la transaction (USD)
    TransactionAmt      DECIMAL(12, 2),

    -- Type de carte de paiement
    ProductCD           VARCHAR(10),    -- W, H, C, S, R
    card1               SMALLINT,       -- numéro carte (masqué)
    card2               SMALLINT,
    card3               SMALLINT,
    card4               VARCHAR(20),    -- visa, mastercard, etc.
    card5               SMALLINT,
    card6               VARCHAR(20),    -- debit, credit

    -- Adresse
    addr1               SMALLINT,       -- code postal (masqué)
    addr2               SMALLINT,       -- pays (masqué)

    -- Distance adresse de facturation / livraison
    dist1               FLOAT,
    dist2               FLOAT,

    -- Adresses email (domaines)
    P_emaildomain       VARCHAR(100),   -- payeur
    R_emaildomain       VARCHAR(100),   -- destinataire

    -- Comptages agrégés (par carte, adresse…)
    C1  FLOAT, C2  FLOAT, C3  FLOAT, C4  FLOAT,
    C5  FLOAT, C6  FLOAT, C7  FLOAT, C8  FLOAT,
    C9  FLOAT, C10 FLOAT, C11 FLOAT, C12 FLOAT,
    C13 FLOAT, C14 FLOAT,

    -- Délais (temps écoulé depuis dernier événement, en jours)
    D1  FLOAT, D2  FLOAT, D3  FLOAT,  D4  FLOAT,
    D5  FLOAT, D6  FLOAT, D7  FLOAT,  D8  FLOAT,
    D9  FLOAT, D10 FLOAT, D11 FLOAT,  D12 FLOAT,
    D13 FLOAT, D14 FLOAT, D15 FLOAT,

    -- Correspondances booléennes (match adresse, email, etc.)
    M1 VARCHAR(5), M2 VARCHAR(5), M3 VARCHAR(5),
    M4 VARCHAR(5), M5 VARCHAR(5), M6 VARCHAR(5),
    M7 VARCHAR(5), M8 VARCHAR(5), M9 VARCHAR(5),

    -- Variables de comptage Vxxx (comportement de la carte)
    V1  FLOAT, V2  FLOAT, V3  FLOAT, V4  FLOAT, V5  FLOAT,
    V6  FLOAT, V7  FLOAT, V8  FLOAT, V9  FLOAT, V10 FLOAT,
    V11 FLOAT, V12 FLOAT, V13 FLOAT, V14 FLOAT, V15 FLOAT,
    V16 FLOAT, V17 FLOAT, V18 FLOAT, V19 FLOAT, V20 FLOAT,
    V21 FLOAT, V22 FLOAT, V23 FLOAT, V24 FLOAT, V25 FLOAT,
    V26 FLOAT, V27 FLOAT, V28 FLOAT, V29 FLOAT, V30 FLOAT,
    V31 FLOAT, V32 FLOAT, V33 FLOAT, V34 FLOAT, V35 FLOAT,
    V36 FLOAT, V37 FLOAT, V38 FLOAT, V39 FLOAT, V40 FLOAT,
    V41 FLOAT, V42 FLOAT, V43 FLOAT, V44 FLOAT, V45 FLOAT,
    V46 FLOAT, V47 FLOAT, V48 FLOAT, V49 FLOAT, V50 FLOAT,
    V51 FLOAT, V52 FLOAT, V53 FLOAT, V54 FLOAT, V55 FLOAT,
    V56 FLOAT, V57 FLOAT, V58 FLOAT, V59 FLOAT, V60 FLOAT,
    V61 FLOAT, V62 FLOAT, V63 FLOAT, V64 FLOAT, V65 FLOAT,
    V66 FLOAT, V67 FLOAT, V68 FLOAT, V69 FLOAT, V70 FLOAT,
    V71 FLOAT, V72 FLOAT, V73 FLOAT, V74 FLOAT, V75 FLOAT,
    V76 FLOAT, V77 FLOAT, V78 FLOAT, V79 FLOAT, V80 FLOAT,
    V81 FLOAT, V82 FLOAT, V83 FLOAT, V84 FLOAT, V85 FLOAT,
    V86 FLOAT, V87 FLOAT, V88 FLOAT, V89 FLOAT, V90 FLOAT,
    V91 FLOAT, V92 FLOAT, V93 FLOAT, V94 FLOAT,

    PRIMARY KEY (TransactionID),
    INDEX idx_fraud    (isFraud),
    INDEX idx_dt       (TransactionDT),
    INDEX idx_card1    (card1),
    INDEX idx_amt      (TransactionAmt),
    INDEX idx_product  (ProductCD)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 2 : identity
-- Source : train_identity.csv (144 233 lignes)
-- Rôle   : informations device / navigateur de la transaction
-- Jointure : LEFT JOIN sur TransactionID (pas toujours disponible)
-- ============================================================
DROP TABLE IF EXISTS identity;

CREATE TABLE identity (
    TransactionID       BIGINT          NOT NULL,

    -- Type d'appareil et information navigateur
    DeviceType          VARCHAR(20),    -- desktop / mobile
    DeviceInfo          VARCHAR(200),   -- modèle exact (ex: "Windows")

    -- Variables d'identité (navigateur, réseau, etc.)
    id_01 FLOAT, id_02 FLOAT, id_03 FLOAT, id_04 FLOAT,
    id_05 FLOAT, id_06 FLOAT, id_07 FLOAT, id_08 FLOAT,
    id_09 FLOAT, id_10 FLOAT, id_11 FLOAT,

    id_12 VARCHAR(20),  id_13 FLOAT,   id_14 FLOAT,
    id_15 VARCHAR(20),  id_16 VARCHAR(20),
    id_17 FLOAT,        id_18 FLOAT,   id_19 FLOAT,
    id_20 FLOAT,        id_21 FLOAT,   id_22 FLOAT,
    id_23 VARCHAR(50),  id_24 FLOAT,   id_25 FLOAT,
    id_26 FLOAT,        id_27 VARCHAR(10), id_28 VARCHAR(10),
    id_29 VARCHAR(10),  id_30 VARCHAR(50), id_31 VARCHAR(100),
    id_32 FLOAT,        id_33 VARCHAR(20), id_34 VARCHAR(10),
    id_35 VARCHAR(10),  id_36 VARCHAR(10), id_37 VARCHAR(10),
    id_38 VARCHAR(10),

    PRIMARY KEY (TransactionID),
    FOREIGN KEY (TransactionID)
        REFERENCES transactions(TransactionID)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;


-- ============================================================
-- TABLE 3 : fraud_predictions
-- Rôle : stocker les prédictions du modèle LightGBM
--        pour monitoring et audit des alertes
-- ============================================================
DROP TABLE IF EXISTS fraud_predictions;

CREATE TABLE fraud_predictions (
    prediction_id       BIGINT          NOT NULL AUTO_INCREMENT,
    TransactionID       BIGINT          NOT NULL,

    -- Sortie du modèle
    fraud_score         FLOAT           NOT NULL,   -- probabilité [0, 1]
    is_fraud_predicted  TINYINT(1)      NOT NULL,   -- 1 si score > threshold
    threshold_used      FLOAT           NOT NULL DEFAULT 0.5,

    -- Vérité terrain (disponible après investigation)
    is_fraud_actual     TINYINT(1),                 -- NULL jusqu'à résolution
    investigation_status VARCHAR(20) DEFAULT 'PENDING',
    -- PENDING | CONFIRMED_FRAUD | FALSE_POSITIVE | CLEARED

    -- Métadonnées
    model_version       VARCHAR(50)     NOT NULL DEFAULT 'lgbm_v1',
    predicted_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at         TIMESTAMP,

    PRIMARY KEY (prediction_id),
    INDEX idx_txid      (TransactionID),
    INDEX idx_score     (fraud_score DESC),
    INDEX idx_status    (investigation_status),
    FOREIGN KEY (TransactionID)
        REFERENCES transactions(TransactionID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

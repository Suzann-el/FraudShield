-- ============================================================
-- FraudShield — Import des données CSV
-- ============================================================
-- PRÉREQUIS :
--   1. Copier les CSV dans le répertoire secure_file_priv :
--      SHOW VARIABLES LIKE 'secure_file_priv';
--      (souvent : C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\)
--
--   2. OU utiliser le script Python (recommandé) :
--      python src/import_data.py
-- ============================================================

USE fraudshield;

-- ── Import transactions ──────────────────────────────────────
-- Adapter le chemin si nécessaire
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/train_transaction.csv'
INTO TABLE transactions
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(
    TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD,
    card1, card2, card3, card4, card5, card6,
    addr1, addr2, dist1, dist2,
    P_emaildomain, R_emaildomain,
    C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,
    D1,D2,D3,D4,D5,D6,D7,D8,D9,D10,D11,D12,D13,D14,D15,
    M1,M2,M3,M4,M5,M6,M7,M8,M9,
    V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,
    V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,
    V21,V22,V23,V24,V25,V26,V27,V28,V29,V30,
    V31,V32,V33,V34,V35,V36,V37,V38,V39,V40,
    V41,V42,V43,V44,V45,V46,V47,V48,V49,V50,
    V51,V52,V53,V54,V55,V56,V57,V58,V59,V60,
    V61,V62,V63,V64,V65,V66,V67,V68,V69,V70,
    V71,V72,V73,V74,V75,V76,V77,V78,V79,V80,
    V81,V82,V83,V84,V85,V86,V87,V88,V89,V90,
    V91,V92,V93,V94
);

SELECT CONCAT('Transactions importées : ', COUNT(*)) AS resultat FROM transactions;

-- ── Import identity ──────────────────────────────────────────
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/train_identity.csv'
INTO TABLE identity
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(
    TransactionID,
    id_01,id_02,id_03,id_04,id_05,id_06,id_07,id_08,id_09,id_10,id_11,
    id_12,id_13,id_14,id_15,id_16,id_17,id_18,id_19,id_20,
    id_21,id_22,id_23,id_24,id_25,id_26,id_27,id_28,id_29,id_30,
    id_31,id_32,id_33,id_34,id_35,id_36,id_37,id_38,
    DeviceType, DeviceInfo
);

SELECT CONCAT('Identités importées : ', COUNT(*)) AS resultat FROM identity;

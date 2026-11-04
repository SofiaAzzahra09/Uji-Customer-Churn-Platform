-- database/schema.sql
-- Customer Churn Prediction System - Model Registry Schema

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    versi TEXT NOT NULL UNIQUE,
    tanggal_training TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    jumlah_data INTEGER NOT NULL,
    dataset_filename TEXT,

    -- Metrik Evaluasi (Test Set)
    roc_auc REAL NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    brier_score REAL,
    optimal_threshold REAL,

    -- Confusion Matrix
    tn INTEGER,
    fp INTEGER,
    fn INTEGER,
    tp INTEGER,

    -- Overfitting Detection
    train_roc_auc REAL,
    test_roc_auc REAL,
    model_status TEXT,

    -- Training Info
    n_features INTEGER,
    hyperparameter TEXT NOT NULL,
    file_path TEXT NOT NULL,
    metadata_path TEXT,

    -- Status & Catatan
    status TEXT DEFAULT 'nonaktif',
    catatan TEXT
);

CREATE INDEX IF NOT EXISTS idx_model_registry_versi
ON model_registry(versi);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
ON model_registry(status);

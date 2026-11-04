"""
================================================================================
CUSTOMER CHURN PREDICTION SYSTEM USING CRISP-DM METHODOLOGY
================================================================================
Author: Sofia Azzahra Alamsyah
Methodology: CRISP-DM (Single Model: Logistic Regression + Class Weight Balanced)
Features:
  - Data Cleaning (Duplikasi, Inkonsistensi, Missing Values, Outlier Capping)
  - Feature Engineering (Fitur Turunan, Ordinal + One-Hot Encoding)
  - Modeling (Logistic Regression + GridSearchCV + 5-Fold Stratified CV)
  - Evaluation (ROC-AUC, Threshold Optimization, Overfitting Detection)
  - Deployment (SQLite Model Registry + Dynamic Metadata JSON)
================================================================================
"""

import os
import sys
import json
import joblib
import pandas as pd

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from services.db_service import DatabaseService
from services.model_registry import ModelRegistry
from services.training_service import TrainingService


def main():
    print("=" * 70)
    print("PHASE 1: BUSINESS UNDERSTANDING")
    print("=" * 70)
    print("Tujuan: Memprediksi pelanggan yang berpotensi churn (berhenti berlangganan)")
    print("untuk merancang strategi retensi dini dan menekan Customer Acquisition Cost.")
    print("Pendekatan: Random Forest Classifier + SMOTE + GridSearchCV.\n")

    # --------------------------------------------------------------------------
    # PHASE 2: DATA UNDERSTANDING
    # --------------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 2: DATA UNDERSTANDING")
    print("=" * 70)
    dataset_path = os.path.join(BASE_DIR, "Raw dataset", "ml_portfolio_customer_churn_dataset.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"Dimensi Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"\nDistribusi Target (Churn):")
    churn_counts = df['churn'].value_counts()
    churn_pct = df['churn'].value_counts(normalize=True) * 100
    print(f"  - Stayed (0): {churn_counts.get(0, 0)} ({churn_pct.get(0, 0):.2f}%)")
    print(f"  - Churned (1): {churn_counts.get(1, 0)} ({churn_pct.get(1, 0):.2f}%) -> Class Imbalance\n")

    # --------------------------------------------------------------------------
    # PHASE 3-5: TRAINING (via TrainingService)
    # --------------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 3-5: DATA PREPARATION + MODELING + EVALUATION")
    print("=" * 70)

    trainer_svc = TrainingService()
    pipeline, metrics, best_params = trainer_svc.train(
        df,
        catatan="Initial training dari dataset asli (SMOTE + Threshold Optimization)",
        dataset_filename="ml_portfolio_customer_churn_dataset.csv"
    )

    # Print cleaning log
    print("\n--- Data Cleaning Log ---")
    for entry in metrics.get('cleaning_log', []):
        print(f"  {entry}")

    # Print evaluation
    print(f"\n--- Evaluasi Model (Test Set: {metrics['n_test_samples']} sampel) ---")
    print(f"  Model Status     : {metrics['model_status']}")
    print(f"  Train ROC-AUC    : {metrics['train_roc_auc']:.4f}")
    print(f"  Test ROC-AUC     : {metrics['test_roc_auc']:.4f}")
    print(f"  CV ROC-AUC       : {metrics['cv_roc_auc']:.4f}")
    print(f"  Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"  Accuracy         : {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Precision (Churn): {metrics['test_precision'] * 100:.2f}%")
    print(f"  Recall (Churn)   : {metrics['test_recall'] * 100:.2f}%")
    print(f"  F1-Score (Churn) : {metrics['test_f1']:.4f}")
    print(f"  Brier Score      : {metrics['test_brier']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"  Confusion Matrix : TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}")

    # --------------------------------------------------------------------------
    # PHASE 6: DEPLOYMENT (Save to SQLite Registry)
    # --------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("PHASE 6: DEPLOYMENT (SQLite Model Registry)")
    print("=" * 70)

    db_svc = DatabaseService()
    reg = ModelRegistry(db_svc)

    versi = reg.save(
        pipeline,
        metrics,
        best_params,
        catatan="Initial training - SMOTE + Threshold Optimization",
        dataset_filename="ml_portfolio_customer_churn_dataset.csv"
    )

    # Also save as current active model files for backward compat
    joblib.dump(pipeline, os.path.join(BASE_DIR, "models", "best_churn_model.pkl"))

    meta_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'versi': versi,
            'model_name': f'Random Forest Classifier ({versi})',
            'algorithm': 'Random Forest',
            'model_status': metrics['model_status'],
            'cv_roc_auc': metrics['cv_roc_auc'],
            'optimal_threshold': metrics['optimal_threshold'],
            'best_hyperparameters': best_params,
            'test_metrics': metrics['test_metrics'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report'],
            'roc_curve': metrics['roc_curve'],
            'pr_curve': metrics['pr_curve'],
            'feature_importance': metrics['feature_importance'],
            'n_training_samples': metrics['n_training_samples'],
            'n_test_samples': metrics['n_test_samples'],
            'cleaning_log': metrics['cleaning_log'],
            'train_roc_auc': metrics['train_roc_auc'],
            'test_roc_auc': metrics['test_roc_auc'],
        }, f, indent=4, default=str)

    print(f"Model {versi} berhasil disimpan ke registry SQLite!")
    print(f"Status: AKTIF")
    print(f"\nFile artifacts:")
    print(f"  - Pipeline: models/churn_model_{versi}.pkl")
    print(f"  - Metadata: models/metadata_{versi}.json")
    print(f"  - Database: database/churn_system.db")
    print(f"\nProses CRISP-DM selesai! Jalankan 'python app.py' untuk dashboard.")


if __name__ == '__main__':
    main()

# services/training_service.py
"""
Training Service — Modular CRISP-DM pipeline for Customer Churn Prediction.
Includes: data cleaning, inconsistency handling, outlier capping, feature
engineering, Logistic Regression with GridSearchCV, threshold optimization,
and overfitting detection.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score
)


class TrainingService:

    VALID_CONTRACTS = {'Monthly', 'Quarterly', 'Annual'}
    VALID_PAYMENTS = {'Credit Card', 'Bank Transfer', 'E-Wallet'}
    REQUIRED_COLUMNS = [
        'customer_id', 'age', 'income_million_idr', 'tenure_months',
        'monthly_spend_idr', 'support_tickets', 'login_frequency_monthly',
        'contract_type', 'payment_method', 'discount_used',
        'satisfaction_score', 'churn'
    ]

    def validate_columns(self, df: pd.DataFrame) -> list:
        """Check that all required columns exist. Returns list of missing columns."""
        return [c for c in self.REQUIRED_COLUMNS if c not in df.columns]

    # =========================================================================
    # PHASE 3A: DATA CLEANING — Duplicates, Inconsistency, Missing, Outliers
    # =========================================================================

    def clean_data(self, df: pd.DataFrame) -> tuple:
        """
        Full data cleaning pipeline. Returns (cleaned_df, cleaning_log).
        """
        log = []
        df = df.copy()
        n_start = len(df)

        # --- 1. Duplikasi ---
        n_dup = df.duplicated().sum()
        if n_dup > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            log.append(f"Duplikasi: {n_dup} baris dihapus")
        else:
            log.append("Duplikasi: 0 baris duplikat ditemukan")

        # --- 2. Inkonsistensi Data ---

        # 2a. Nilai negatif → clip ke 0
        num_cols_nonneg = [
            'age', 'income_million_idr', 'tenure_months',
            'monthly_spend_idr', 'support_tickets',
            'login_frequency_monthly', 'satisfaction_score'
        ]
        for col in num_cols_nonneg:
            if col in df.columns:
                n_neg = (df[col] < 0).sum()
                if n_neg > 0:
                    df[col] = df[col].clip(lower=0)
                    log.append(f"Inkonsistensi: {n_neg} nilai negatif di '{col}' di-clip ke 0")

        # 2b. Nilai di luar rentang valid → clip
        if 'age' in df.columns:
            n_fix = ((df['age'] < 18) | (df['age'] > 100)).sum()
            if n_fix > 0:
                df['age'] = df['age'].clip(18, 100)
                log.append(f"Inkonsistensi: {n_fix} nilai 'age' di-clip ke rentang 18-100")

        if 'satisfaction_score' in df.columns:
            n_fix = ((df['satisfaction_score'] < 1) | (df['satisfaction_score'] > 5)).sum()
            if n_fix > 0:
                df['satisfaction_score'] = df['satisfaction_score'].clip(1.0, 5.0)
                log.append(f"Inkonsistensi: {n_fix} nilai 'satisfaction_score' di-clip ke 1.0-5.0")

        if 'discount_used' in df.columns:
            n_fix = (~df['discount_used'].isin([0, 1])).sum()
            if n_fix > 0:
                df['discount_used'] = df['discount_used'].apply(lambda x: 1 if x > 0 else 0)
                log.append(f"Inkonsistensi: {n_fix} nilai 'discount_used' dinormalisasi ke 0/1")

        if 'churn' in df.columns:
            n_fix = (~df['churn'].isin([0, 1])).sum()
            if n_fix > 0:
                df['churn'] = df['churn'].apply(lambda x: 1 if x > 0 else 0)
                log.append(f"Inkonsistensi: {n_fix} nilai 'churn' dinormalisasi ke 0/1")

        # 2c. Inkonsistensi kategorikal
        if 'contract_type' in df.columns:
            df['contract_type'] = df['contract_type'].astype(str).str.strip().str.title()
            n_invalid = (~df['contract_type'].isin(self.VALID_CONTRACTS)).sum()
            if n_invalid > 0:
                df.loc[~df['contract_type'].isin(self.VALID_CONTRACTS), 'contract_type'] = 'Monthly'
                log.append(f"Inkonsistensi: {n_invalid} nilai 'contract_type' tidak valid → default 'Monthly'")

        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].astype(str).str.strip().str.title()
            n_invalid = (~df['payment_method'].isin(self.VALID_PAYMENTS)).sum()
            if n_invalid > 0:
                df.loc[~df['payment_method'].isin(self.VALID_PAYMENTS), 'payment_method'] = 'Bank Transfer'
                log.append(f"Inkonsistensi: {n_invalid} nilai 'payment_method' tidak valid → default 'Bank Transfer'")

        # 2d. Kontradiksi logis
        if 'tenure_months' in df.columns and 'login_frequency_monthly' in df.columns:
            mask = (df['tenure_months'] == 0) & (df['login_frequency_monthly'] > 0)
            n_fix = mask.sum()
            if n_fix > 0:
                df.loc[mask, 'tenure_months'] = 1
                log.append(f"Kontradiksi: {n_fix} baris tenure=0 tapi login>0 → tenure diset ke 1")

        # --- 3. Missing Value Imputation (Median untuk numerik) ---
        num_missing_cols = ['income_million_idr', 'monthly_spend_idr', 'satisfaction_score']
        for col in num_missing_cols:
            if col in df.columns:
                n_miss = df[col].isnull().sum()
                if n_miss > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    log.append(f"Missing Value: {n_miss} nilai kosong di '{col}' diisi median ({median_val:.2f})")

        # Imputasi kategorikal juga (mode)
        cat_cols = ['contract_type', 'payment_method']
        for col in cat_cols:
            if col in df.columns:
                n_miss = df[col].isnull().sum()
                if n_miss > 0:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    log.append(f"Missing Value: {n_miss} nilai kosong di '{col}' diisi modus ('{mode_val}')")

        # --- 4. Outlier Capping (IQR Winsorizing) ---
        outlier_cols = [
            'age', 'income_million_idr', 'tenure_months',
            'monthly_spend_idr', 'support_tickets',
            'login_frequency_monthly', 'satisfaction_score'
        ]
        for col in outlier_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_out = ((df[col] < lower) | (df[col] > upper)).sum()
                if n_out > 0:
                    df[col] = df[col].clip(lower, upper)
                    log.append(f"Outlier: {n_out} outlier di '{col}' di-cap ke [{lower:.2f}, {upper:.2f}]")

        n_end = len(df)
        log.append(f"Data bersih: {n_end} baris (dari {n_start} awal)")

        return df, log

    # =========================================================================
    # PHASE 3B: FEATURE ENGINEERING
    # =========================================================================

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        df = df.copy()

        # Ordinal mapping untuk durasi kontrak
        contract_map = {'Monthly': 1, 'Quarterly': 3, 'Annual': 12}
        df['contract_duration_months'] = df['contract_type'].map(contract_map).fillna(1)

        # Rasio keluhan per bulan masa langganan
        df['support_tickets_per_tenure'] = df['support_tickets'] / (df['tenure_months'] + 1)

        # Nilai pengeluaran per sesi login
        df['spend_per_login'] = df['monthly_spend_idr'] / (df['login_frequency_monthly'] + 1)

        # Rasio keluhan terhadap frekuensi login
        df['support_to_login_ratio'] = df['support_tickets'] / (df['login_frequency_monthly'] + 1)

        # Rasio pengeluaran terhadap pendapatan (satuan konsisten: ribuan / ribuan)
        df['spend_to_income_ratio'] = (
            df['monthly_spend_idr'] / (df['income_million_idr'] * 1000 + 1)
        ).clip(0, 5)

        # Flag kepuasan kritis
        df['is_low_satisfaction'] = (df['satisfaction_score'] <= 2.5).astype(int)

        # Flag support tinggi
        df['high_support_flag'] = (df['support_tickets'] >= 3).astype(int)

        # Segmentasi tenure
        df['tenure_segment'] = df['tenure_months'].apply(
            lambda m: 'New' if m <= 12 else ('Medium' if m <= 36 else 'Loyal')
        )

        return df

    # =========================================================================
    # PHASE 4: BUILD PIPELINE & TRAIN
    # =========================================================================

    def build_pipeline(self):
        """Construct the preprocessing + classifier pipeline."""

        numeric_features = [
            'age', 'income_million_idr', 'tenure_months', 'monthly_spend_idr',
            'support_tickets', 'login_frequency_monthly', 'satisfaction_score',
            'contract_duration_months', 'support_tickets_per_tenure',
            'spend_per_login', 'is_low_satisfaction', 'high_support_flag',
            'discount_used'
        ]
        categorical_features = ['payment_method', 'tenure_segment']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Standard sklearn Pipeline — LogisticRegression for interpretability
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ))
        ])

        return pipeline, numeric_features, categorical_features

    def train(self, df: pd.DataFrame, catatan: str = "",
              dataset_filename: str = "") -> tuple:
        """
        Full training pipeline. Returns (pipeline, metrics, best_params).
        """
        # Clean data
        df_clean, cleaning_log = self.clean_data(df)

        # Feature engineering
        df_fe = self.feature_engineering(df_clean)

        # Prepare X, y
        X = df_fe.drop(columns=['customer_id', 'churn'], errors='ignore')
        y = df_fe['churn']

        # Stratified split 80:20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Build pipeline
        pipeline, numeric_features, categorical_features = self.build_pipeline()

        # Hyperparameter grid for LogisticRegression
        param_grid = {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )

        print("Training Logistic Regression with GridSearchCV (class_weight='balanced')...")
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"  Best Params: {best_params}")

        # Evaluate
        metrics = self.evaluate(
            best_pipeline, X_train, y_train, X_test, y_test,
            numeric_features, categorical_features,
            grid_search.best_score_
        )
        metrics['n_training_samples'] = len(X_train)
        metrics['n_test_samples'] = len(X_test)
        metrics['n_total_data'] = len(df_clean)
        metrics['cleaning_log'] = cleaning_log
        metrics['cv_roc_auc'] = round(float(grid_search.best_score_), 4)
        metrics['best_hyperparameters'] = best_params
        metrics['dataset_filename'] = dataset_filename

        return best_pipeline, metrics, best_params

    # =========================================================================
    # PHASE 5: EVALUATION
    # =========================================================================

    def evaluate(self, pipeline, X_train, y_train, X_test, y_test,
                 numeric_features, categorical_features,
                 cv_roc_auc: float) -> dict:
        """Comprehensive evaluation with threshold optimization & overfitting detection."""

        y_pred_default = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]

        # --- Optimal Threshold via Youden's J Statistic (Balanced Sensitivity & Specificity) ---
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = float(roc_thresholds[best_idx]) if best_idx < len(roc_thresholds) else 0.5
        optimal_threshold = max(0.30, min(0.70, optimal_threshold))  # Bound to reasonable enterprise operational threshold

        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)

        # Metrics with optimal threshold
        test_roc_auc = float(roc_auc_score(y_test, y_prob))
        train_roc_auc = float(roc_auc_score(y_train, y_prob_train))
        test_accuracy = float(accuracy_score(y_test, y_pred_optimal))
        test_precision = float(precision_score(y_test, y_pred_optimal, zero_division=0))
        test_recall = float(recall_score(y_test, y_pred_optimal, zero_division=0))
        test_f1 = float(f1_score(y_test, y_pred_optimal, zero_division=0))
        test_brier = float(brier_score_loss(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred_optimal)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        report_dict = classification_report(y_test, y_pred_optimal, output_dict=True, zero_division=0)

        # Overfitting detection (inspired by Creatroka)
        model_status = self.detect_model_status(train_roc_auc, test_roc_auc)

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_points = [
            {'fpr': round(float(f), 4), 'tpr': round(float(t), 4)}
            for f, t in zip(fpr, tpr)
        ]

        # PR Curve points
        precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
        pr_points = [
            {'recall': round(float(r), 4), 'precision': round(float(p), 4)}
            for r, p in zip(recalls, precisions)
        ]

        # Feature importance (using absolute coefficients for LogisticRegression)
        try:
            clf = pipeline.named_steps['classifier']
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            ohe_names = ohe.get_feature_names_out(categorical_features)
            all_features = numeric_features + list(ohe_names)
            # Use absolute coefficient values as importance proxy
            importances = np.abs(clf.coef_[0])
            # Normalize to sum to 1
            importances = importances / (importances.sum() + 1e-10)
            n_features = len(all_features)

            sorted_idx = np.argsort(importances)[::-1]
            feature_importance = [
                {
                    'feature': all_features[i],
                    'importance': round(float(importances[i]), 4),
                    'percentage': round(float(importances[i]) * 100, 2)
                }
                for i in sorted_idx
            ]
        except Exception:
            feature_importance = []
            n_features = 0

        return {
            'test_roc_auc': round(test_roc_auc, 4),
            'train_roc_auc': round(train_roc_auc, 4),
            'test_accuracy': round(test_accuracy, 4),
            'test_precision': round(test_precision, 4),
            'test_recall': round(test_recall, 4),
            'test_f1': round(test_f1, 4),
            'test_brier': round(test_brier, 4),
            'optimal_threshold': round(optimal_threshold, 4),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'model_status': model_status,
            'n_features': n_features,
            'classification_report': report_dict,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'test_metrics': {
                'roc_auc': round(test_roc_auc, 4),
                'accuracy': round(test_accuracy, 4),
                'precision': round(test_precision, 4),
                'recall': round(test_recall, 4),
                'f1_score': round(test_f1, 4),
                'brier_score': round(test_brier, 4),
                'optimal_threshold': round(optimal_threshold, 4),
                'total_test_samples': len(y_test)
            },
            'roc_curve': roc_points,
            'pr_curve': pr_points,
            'feature_importance': feature_importance
        }

    def detect_model_status(self, train_score: float, test_score: float) -> str:
        """Overfitting/Underfitting detection (adapted from Creatroka)."""
        gap = abs(train_score - test_score)

        if train_score > 0.90 and test_score < 0.65 and gap > 0.20:
            return "Overfitting"
        elif train_score < 0.60 and test_score < 0.60:
            return "Underfitting"
        elif train_score > 0.65 and test_score > 0.60 and gap < 0.15:
            return "Good Fit"
        else:
            return "Moderate Fit"

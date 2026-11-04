# 📜 Changelog — Customer Churn Prediction & Retention System

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-11-03 — Enterprise Grade Release 🚀

### Added
- **12-Factor App Configuration (`config.py` & `.env`):** Isolated secrets, paths, and environment settings.
- **OWASP Security Hardening:** Enforced `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `X-XSS-Protection`, and `Referrer-Policy` headers.
- **Audit Trail & Correlation ID:** Integrated `X-Request-ID` across all incoming HTTP requests and responses.
- **Enterprise Health Probe (`/health`):** JSON endpoint reporting system readiness, model loading status, and active model version.
- **Automated Unit Test Suite (`tests/test_churn_app.py`):** 12 comprehensive automated tests covering auth, session handling, single & batch inference, data cleaning, and registry.
- **Multi-Stage Containerization (`Dockerfile` & `docker-compose.yml`):** Lightweight build based on `python:3.11-slim` running under unprivileged user `appuser:1000`.
- **One-File Enterprise Launcher (`run.py`):** Unified CLI launcher supporting `--test`, `--retrain`, and custom port flags.
- **Youden's J Statistic Thresholding:** Calibrated decision threshold in `TrainingService` to balance sensitivity and specificity.
- **Governance & Open Source Docs:** Added `LICENSE` (MIT), `SECURITY.md`, and `CHANGELOG.md`.

---

## [1.0.0] - 2026-08-30 — Initial Academic Release

### Added
- CRISP-DM methodology adaptation for telecommunication/subscription customer churn.
- Flask web interface with 8 HTML views (login, dashboard, predict, batch, evaluation, models, retrain).
- SQLite model registry tracking version lifecycle and evaluation metrics.
- Data cleaning pipeline with IQR outlier capping and missing value imputation.

"""
Centralized Configuration Module
Customer Churn Prediction System
Author: Sofia Azzahra Alamsyah

Prinsip 12-Factor App:
- Memuat konfigurasi dari .env dan environment variables OS.
- Mengisolasi credential (secret_key, admin credentials) dari source code.
- Resolusi path database dan model secara absolut.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env file jika ada
load_dotenv(BASE_DIR / ".env")


class Config:
    """Base application configuration."""

    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    SECRET_KEY = os.getenv("SECRET_KEY", "churn-enterprise-fallback-secret-key-2026")

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "5000"))

    # Database
    DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "database/churn_system.db")

    # Artifacts
    MODEL_PATH = BASE_DIR / os.getenv("MODEL_PATH", "models/best_churn_model.pkl")
    METADATA_PATH = BASE_DIR / os.getenv("METADATA_PATH", "models/model_metadata.json")
    RAW_DATA_PATH = BASE_DIR / os.getenv("RAW_DATA_PATH", "Raw dataset/ml_portfolio_customer_churn_dataset.csv")

    # Security & Auth
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"

    @classmethod
    def init_app(cls, app):
        """Initialise app with configuration."""
        app.config["SECRET_KEY"] = cls.SECRET_KEY
        app.config["DEBUG"] = cls.DEBUG

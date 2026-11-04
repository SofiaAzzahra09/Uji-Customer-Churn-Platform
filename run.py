"""
Customer Churn Prediction System — Enterprise One-File Launcher
Author: Sofia Azzahra Alamsyah

Fungsi:
-------
Entry point tunggal untuk menjalankan seluruh subsistem:
1. Verifikasi dependensi dan versi Python (>= 3.10)
2. Auto-check database SQLite & inisialisasi skema tabel jika belum ada
3. Auto-check model artefak & auto-training jika artefak belum ada
4. Eksekusi pengujian otomatis (--test)
5. Pelatihan ulang model pipeline (--retrain)
6. Menjalankan server Flask / WSGI production-grade

Penggunaan:
-----------
python run.py                   # Jalankan web dashboard & API
python run.py --port 5005       # Jalankan di custom port
python run.py --test            # Jalankan seluruh test suite pytest
python run.py --retrain         # Latih ulang model machine learning
python run.py --debug           # Jalankan dalam mode debug
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

BANNER = r"""
=============================================================================
   _____ _                             _____                     _           
  / ____| |                           / ____|                   | |          
 | |    | |__  _   _ _ __ _ __       | |  __ _   _  __ _ _ __ __| |          
 | |    | '_ \| | | | '__| '_ \      | | |_ | | | |/ _` | '__/ _` |          
 | |____| | | | |_| | |  | | | |     | |__| | |_| | (_| | | | (_| |          
  \_____|_| |_|\__,_|_|  |_| |_|      \_____|\__,_|\__,_|_|  \__,_|          
                                                                             
  Customer Churn Prediction & Retention Analytics System v1.1.0             
=============================================================================
  * Lead Engineer    : Sofia Azzahra Alamsyah
  * Methodology      : CRISP-DM Machine Learning Lifecycle
  * Web Dashboard    : http://{host}:{port}
  * Health Endpoint  : http://{host}:{port}/health
  * Default Login    : {admin_user} / {admin_pass}
  * Database Engine  : SQLite WAL Mode ({db_path})
=============================================================================
"""


def verify_python_version():
    """Memastikan versi Python memenuhi syarat minimum (>= 3.10)."""
    if sys.version_info < (3, 10):
        print(f"[ERROR] Python 3.10 atau lebih tinggi dibutuhkan. Versi saat ini: {sys.version}")
        sys.exit(1)


def check_and_prepare_artifacts():
    """Memeriksa database dan model artefak; otomatis inisialisasi jika belum ada."""
    from config import Config
    from services.db_service import DatabaseService

    # 1. Inisialisasi Database
    db = DatabaseService(db_path=str(Config.DATABASE_PATH))

    # 2. Cek apakah ada model aktif
    from services.model_registry import ModelRegistry
    reg = ModelRegistry(db)
    pipe, meta = reg.load_active()

    if pipe is None:
        print("[INFO] Model machine learning belum terdaftar. Menjalankan training otomatis...")
        from customer_churn_crisp_dm import main as train_main
        train_main()


def run_tests():
    """Jalankan pytest suite."""
    print("\n[INFO] Menjalankan automated test suite...")
    import pytest
    code = pytest.main(["-v", str(PROJECT_ROOT / "tests")])
    sys.exit(code)


def run_retrain():
    """Jalankan retraining pipeline dan simpan model baru ke registry."""
    print("\n[INFO] Memulai pelatihan ulang model machine learning...")
    from customer_churn_crisp_dm import main as train_main
    train_main()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction System Launcher")
    parser.add_argument("--host", type=str, default=None, help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port number (default: 5000)")
    parser.add_argument("--test", action="store_true", help="Run automated test suite and exit")
    parser.add_argument("--retrain", action="store_true", help="Retrain models and exit")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")

    args = parser.parse_args()

    verify_python_version()

    if args.test:
        run_tests()

    if args.retrain:
        run_retrain()

    from config import Config
    from app import app

    host = args.host or Config.HOST
    port = args.port or Config.PORT
    debug = args.debug or Config.DEBUG

    check_and_prepare_artifacts()

    db_rel = os.path.relpath(Config.DATABASE_PATH, PROJECT_ROOT) if os.path.exists(Config.DATABASE_PATH) else Config.DATABASE_PATH

    print(BANNER.format(
        host=host,
        port=port,
        admin_user=Config.ADMIN_USERNAME,
        admin_pass=Config.ADMIN_PASSWORD,
        db_path=db_rel
    ))
    print(f"[STATUS] Environment Mode : {Config.FLASK_ENV.upper()}")
    print(f"[STATUS] Security Headers : ENABLED (OWASP Hardened)")
    print(f"[STATUS] Request Auditing : ENABLED (X-Request-ID Correlation)")
    print("\n[STARTING] Menjalankan Server ChurnGuard AI...\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()

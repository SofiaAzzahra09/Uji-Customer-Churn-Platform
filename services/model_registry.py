# services/model_registry.py
"""
Model Registry — Manages versioning, activation, loading, and deletion of
trained model artifacts. Adapted from Creatroka's model_registry.py for
the Customer Churn Prediction System.
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime


class ModelRegistry:
    def __init__(self, db_service, model_dir: str = None):
        self.db = db_service
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_dir is None:
            self.model_dir = os.path.join(self.base_dir, "models")
        else:
            self.model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(self.base_dir, model_dir)
        os.makedirs(self.model_dir, exist_ok=True)

    def save(self, pipeline, metrics: dict, params: dict,
             catatan: str = "", dataset_filename: str = "") -> str:
        """Save a trained model pipeline + metrics to registry."""
        versi = self._next_version()

        file_path = os.path.join(self.model_dir, f"churn_model_{versi}.pkl")
        meta_path = os.path.join(self.model_dir, f"metadata_{versi}.json")

        # Save pipeline artifact
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline, f)

        # Save metadata JSON
        metadata = {
            'versi': versi,
            'model_name': f'Random Forest Classifier ({versi})',
            'algorithm': 'Random Forest',
            'training_timestamp': datetime.now().isoformat(),
            'dataset_filename': dataset_filename,
            **metrics
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, default=str)

        # Determine initial status
        is_first = self.get_all().empty
        status_awal = "aktif" if is_first else "nonaktif"

        self.db.execute("""
            INSERT INTO model_registry (
                versi, jumlah_data, dataset_filename,
                roc_auc, accuracy, precision_score, recall_score,
                f1_score, brier_score, optimal_threshold,
                tn, fp, fn, tp,
                train_roc_auc, test_roc_auc, model_status,
                n_features, hyperparameter,
                file_path, metadata_path,
                status, catatan
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            versi,
            metrics.get('n_training_samples', 0),
            dataset_filename,
            metrics.get('test_roc_auc', 0),
            metrics.get('test_accuracy', 0),
            metrics.get('test_precision', 0),
            metrics.get('test_recall', 0),
            metrics.get('test_f1', 0),
            metrics.get('test_brier', 0),
            metrics.get('optimal_threshold', 0.5),
            metrics.get('tn', 0),
            metrics.get('fp', 0),
            metrics.get('fn', 0),
            metrics.get('tp', 0),
            metrics.get('train_roc_auc', 0),
            metrics.get('test_roc_auc', 0),
            metrics.get('model_status', 'Unknown'),
            metrics.get('n_features', 0),
            json.dumps(params),
            file_path,
            meta_path,
            status_awal,
            catatan
        ))

        return versi

    def activate(self, versi: str) -> bool:
        """Set a specific model version as active (deactivate all others)."""
        self.db.execute("UPDATE model_registry SET status='nonaktif'")
        self.db.execute(
            "UPDATE model_registry SET status='aktif' WHERE versi=?",
            (versi,)
        )
        return True

    def load_active(self):
        """Load the active model pipeline from disk."""
        row = self.db.fetchone(
            "SELECT file_path, metadata_path FROM model_registry WHERE status='aktif'"
        )
        if not row:
            return None, {}

        pipeline = None
        metadata = {}

        fpath = row['file_path']
        if fpath and not os.path.isabs(fpath):
            fpath = os.path.join(self.base_dir, fpath)

        mpath = row['metadata_path']
        if mpath and not os.path.isabs(mpath):
            mpath = os.path.join(self.base_dir, mpath)

        if fpath and os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                pipeline = pickle.load(f)

        if mpath and os.path.exists(mpath):
            with open(mpath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        return pipeline, metadata

    def get_active_info(self) -> dict:
        """Get the active model's DB row as a dict."""
        return self.db.fetchone(
            "SELECT * FROM model_registry WHERE status='aktif'"
        )

    def get_all(self) -> pd.DataFrame:
        """Get all models as a DataFrame, sorted newest first."""
        return self.db.fetchdf(
            "SELECT * FROM model_registry ORDER BY id DESC"
        )

    def delete(self, versi: str):
        """Delete a model version (cannot delete active model)."""
        row = self.db.fetchone(
            "SELECT status, file_path, metadata_path FROM model_registry WHERE versi=?",
            (versi,)
        )
        if not row:
            raise ValueError("Model tidak ditemukan.")
        if row['status'].strip().lower() == 'aktif':
            raise ValueError("Tidak bisa menghapus model yang sedang aktif.")

        # Remove files
        if os.path.exists(row['file_path']):
            os.remove(row['file_path'])
        if row['metadata_path'] and os.path.exists(row['metadata_path']):
            os.remove(row['metadata_path'])

        self.db.execute(
            "DELETE FROM model_registry WHERE versi=?", (versi,)
        )

    def _next_version(self) -> str:
        row = self.db.fetchone(
            "SELECT MAX(CAST(SUBSTR(versi,2) AS INTEGER)) AS n FROM model_registry"
        )
        n = (row['n'] or 0) + 1 if row else 1
        return f"v{n}"

# services/db_service.py
"""
Thin SQLite wrapper for the Customer Churn Prediction System.
Auto-creates tables from database/schema.sql on initialization.
"""

import os
import sqlite3
import pandas as pd


class DatabaseService:
    def __init__(self, db_path: str = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "database", "churn_system.db")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        schema_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "database", "schema.sql"
        )
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            conn = self._get_conn()
            try:
                conn.executescript(schema_sql)
                conn.commit()
            finally:
                conn.close()

    def execute(self, sql: str, params: tuple = ()):
        conn = self._get_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def fetchone(self, sql: str, params: tuple = ()) -> dict:
        conn = self._get_conn()
        try:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def fetchdf(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        conn = self._get_conn()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        finally:
            conn.close()

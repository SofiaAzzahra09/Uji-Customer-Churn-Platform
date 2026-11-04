"""
Enterprise Test Suite for Customer Churn Prediction System
Author: Sofia Azzahra Alamsyah
"""

import pytest
import io
import pandas as pd
from app import app
from config import Config
from services.training_service import TrainingService


@pytest.fixture
def client():
    """Create Flask test client with test configuration."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as test_client:
        yield test_client


def login_admin(client):
    """Helper to log in as admin user."""
    return client.post("/login", data={
        "username": Config.ADMIN_USERNAME,
        "password": Config.ADMIN_PASSWORD,
    }, follow_redirects=True)


# ── 1. Infrastructure & Health ─────────────────────────────────────────
def test_health_check_endpoint(client):
    """Verify health endpoint returns 200, healthy status, and active model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "ChurnGuard" in data["service"]


def test_security_headers_and_request_id(client):
    """Verify OWASP security headers and correlation ID."""
    response = client.get("/login", headers={"X-Request-ID": "test-uuid-1234"})
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert response.headers.get("X-Request-ID") == "test-uuid-1234"


# ── 2. Authentication & Session Security ──────────────────────────────
def test_login_page_renders(client):
    """Verify login page renders correctly."""
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Login" in response.data or b"Masuk" in response.data


def test_unauthenticated_access_redirects_to_login(client):
    """Verify protected routes redirect unauthenticated users to /login."""
    protected_urls = ["/", "/predict", "/batch", "/evaluation", "/models", "/retrain"]
    for url in protected_urls:
        response = client.get(url, follow_redirects=False)
        assert response.status_code in [302, 308], f"URL {url} did not redirect"
        assert "/login" in response.headers.get("Location", "")


def test_admin_successful_login(client):
    """Verify valid admin login establishes session and redirects to dashboard."""
    response = login_admin(client)
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert "user" in sess
        assert sess["user"]["username"] == Config.ADMIN_USERNAME


def test_admin_invalid_login(client):
    """Verify wrong password fails and user is not authenticated."""
    response = client.post("/login", data={
        "username": Config.ADMIN_USERNAME,
        "password": "wrong_password_123",
    }, follow_redirects=True)
    assert response.status_code == 200
    with client.session_transaction() as sess:
        assert "user" not in sess


# ── 3. Authenticated Dashboard & Features ──────────────────────────────
def test_authenticated_dashboard_access(client):
    """Verify authenticated dashboard loads successfully."""
    login_admin(client)
    response = client.get("/")
    assert response.status_code == 200
    assert b"Churn" in response.data or b"Dashboard" in response.data


def test_single_prediction_form_get(client):
    """Verify single prediction form renders for logged-in user."""
    login_admin(client)
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Prediksi" in response.data or b"customer" in response.data.lower()


def test_single_prediction_post(client):
    """Verify single prediction calculation produces probability and retention advice."""
    login_admin(client)
    payload = {
        "customer_id": "CUST-TEST-001",
        "age": "45",
        "income_million_idr": "7.5",
        "tenure_months": "6",
        "monthly_spend_idr": "600",
        "support_tickets": "4",
        "login_frequency_monthly": "3",
        "contract_type": "Monthly",
        "payment_method": "E-Wallet",
        "discount_used": "0",
        "satisfaction_score": "2.0",
    }
    response = client.post("/predict", data=payload, follow_redirects=True)
    assert response.status_code == 200
    # Should display high churn probability or risk badge
    assert (
        b"Tinggi" in response.data
        or b"Probabilitas" in response.data
        or b"High" in response.data
        or b"Hasil Prediksi" in response.data
    )


def test_evaluation_page_displays_metrics(client):
    """Verify evaluation page displays model ROC-AUC and metrics."""
    login_admin(client)
    response = client.get("/evaluation")
    assert response.status_code == 200
    assert b"ROC-AUC" in response.data


def test_models_registry_page(client):
    """Verify models registry page lists model versions."""
    login_admin(client)
    response = client.get("/models")
    assert response.status_code == 200
    assert b"v2" in response.data or b"v1" in response.data


# ── 4. Training Service Unit Tests ────────────────────────────────────
def test_data_cleaning_service():
    """Verify data cleaning service handles duplicates, negative numbers, and nulls."""
    trainer = TrainingService()
    raw_df = pd.DataFrame([
        {
            "customer_id": "C1", "age": -5, "income_million_idr": None,
            "tenure_months": 12, "monthly_spend_idr": 500.0, "support_tickets": -1,
            "login_frequency_monthly": 10.0, "contract_type": "monthly",
            "payment_method": "bank transfer", "discount_used": 1,
            "satisfaction_score": 6.0, "churn": 0
        },
        {
            "customer_id": "C1", "age": -5, "income_million_idr": None,
            "tenure_months": 12, "monthly_spend_idr": 500.0, "support_tickets": -1,
            "login_frequency_monthly": 10.0, "contract_type": "monthly",
            "payment_method": "bank transfer", "discount_used": 1,
            "satisfaction_score": 6.0, "churn": 0
        }  # Duplicate row
    ])

    clean_df, log = trainer.clean_data(raw_df)
    # Duplicate removed
    assert len(clean_df) == 1
    # Negative values clipped
    assert clean_df["age"].iloc[0] >= 18
    assert clean_df["support_tickets"].iloc[0] >= 0
    # Out of bound satisfaction clipped
    assert clean_df["satisfaction_score"].iloc[0] <= 5.0
    # Categoricals normalized
    assert clean_df["contract_type"].iloc[0] == "Monthly"
    assert clean_df["payment_method"].iloc[0] == "Bank Transfer"

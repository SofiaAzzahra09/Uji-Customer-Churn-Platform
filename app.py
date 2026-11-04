import os
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from services.db_service import DatabaseService
from services.model_registry import ModelRegistry
from services.training_service import TrainingService

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATASET_PATH = str(Config.RAW_DATA_PATH)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Initialize services
db = DatabaseService(db_path=str(Config.DATABASE_PATH))
registry = ModelRegistry(db)
trainer = TrainingService()

# ── Enterprise Security & Audit Middleware ────────────────────────────
@app.before_request
def assign_request_id():
    """Assign or extract unique correlation ID for audit trail."""
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())


@app.after_request
def set_security_headers(response):
    """Enforce standard OWASP security headers on all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    return response

# Seed Admin Data
SEED_USERS = {
    "admin": {
        "username": "admin",
        "email": "admin@churnguard.ai",
        "name": "Sofia Azzahra",
        "password": generate_password_hash("admin123"),
        "role": "Administrator"
    },
    "analyst": {
        "username": "analyst",
        "email": "analyst@churnguard.ai",
        "name": "Data Analyst Team",
        "password": generate_password_hash("analyst123"),
        "role": "Data Analyst"
    }
}


# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Silakan login terlebih dahulu untuk mengakses menu sistem.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_active_model():
    """Load the active model pipeline and metadata from registry."""
    return registry.load_active()


def preprocess_input(form_data):
    """Compute feature engineering on raw form inputs for prediction."""
    age = float(form_data.get('age', 38))
    income = float(form_data.get('income_million_idr', 8.7))
    tenure = float(form_data.get('tenure_months', 49))
    spend = float(form_data.get('monthly_spend_idr', 450))
    tickets = int(form_data.get('support_tickets', 1))
    logins = float(form_data.get('login_frequency_monthly', 12))
    contract = form_data.get('contract_type', 'Monthly')
    payment = form_data.get('payment_method', 'Credit Card')
    discount = int(form_data.get('discount_used', 0))
    satisfaction = float(form_data.get('satisfaction_score', 3.5))

    contract_map = {'Monthly': 1, 'Quarterly': 3, 'Annual': 12}
    contract_duration_months = contract_map.get(contract, 1)
    support_tickets_per_tenure = tickets / (tenure + 1)
    spend_per_login = spend / (logins + 1)
    support_to_login_ratio = tickets / (logins + 1)
    spend_to_income_ratio = min(spend / (income * 1000 + 1), 5.0)
    is_low_satisfaction = 1 if satisfaction <= 2.5 else 0
    high_support_flag = 1 if tickets >= 3 else 0

    if tenure <= 12:
        tenure_segment = 'New'
    elif tenure <= 36:
        tenure_segment = 'Medium'
    else:
        tenure_segment = 'Loyal'

    input_df = pd.DataFrame([{
        'age': age,
        'income_million_idr': income,
        'tenure_months': tenure,
        'monthly_spend_idr': spend,
        'support_tickets': tickets,
        'login_frequency_monthly': logins,
        'satisfaction_score': satisfaction,
        'contract_duration_months': contract_duration_months,
        'spend_to_income_ratio': spend_to_income_ratio,
        'support_tickets_per_tenure': support_tickets_per_tenure,
        'spend_per_login': spend_per_login,
        'support_to_login_ratio': support_to_login_ratio,
        'is_low_satisfaction': is_low_satisfaction,
        'high_support_flag': high_support_flag,
        'discount_used': discount,
        'contract_type': contract,
        'payment_method': payment,
        'tenure_segment': tenure_segment
    }])

    return input_df, {
        'age': age, 'income': income, 'tenure': tenure, 'spend': spend,
        'tickets': tickets, 'logins': logins, 'contract': contract,
        'payment': payment, 'discount': discount, 'satisfaction': satisfaction,
        'contract_duration_months': contract_duration_months,
        'spend_to_income_ratio': round(spend_to_income_ratio, 4),
        'support_tickets_per_tenure': round(support_tickets_per_tenure, 4),
        'spend_per_login': round(spend_per_login, 2),
        'support_to_login_ratio': round(support_to_login_ratio, 4),
        'is_low_satisfaction': is_low_satisfaction,
        'high_support_flag': high_support_flag,
        'tenure_segment': tenure_segment
    }


def apply_batch_features(df):
    """Apply feature engineering to a batch DataFrame for scoring."""
    contract_map = {'Monthly': 1, 'Quarterly': 3, 'Annual': 12}
    df['contract_duration_months'] = df['contract_type'].map(contract_map).fillna(1)
    df['support_tickets_per_tenure'] = df['support_tickets'] / (df['tenure_months'] + 1)
    df['spend_per_login'] = df['monthly_spend_idr'] / (df['login_frequency_monthly'] + 1)
    df['support_to_login_ratio'] = df['support_tickets'] / (df['login_frequency_monthly'] + 1)
    df['spend_to_income_ratio'] = (df['monthly_spend_idr'] / (df['income_million_idr'] * 1000 + 1)).clip(0, 5)
    df['is_low_satisfaction'] = (df['satisfaction_score'] <= 2.5).astype(int)
    df['high_support_flag'] = (df['support_tickets'] >= 3).astype(int)
    df['tenure_segment'] = df['tenure_months'].apply(
        lambda m: 'New' if m <= 12 else ('Medium' if m <= 36 else 'Loyal')
    )
    return df


# ==================== Auth Routes ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    next_url = request.args.get('next') or request.form.get('next') or url_for('dashboard')

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        user = SEED_USERS.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = {
                'username': user['username'],
                'name': user['name'],
                'email': user['email'],
                'role': user['role']
            }
            flash(f"Selamat datang kembali, {user['name']}! Login berhasil.", "success")
            return redirect(next_url)
        else:
            flash("Username atau password salah. Silakan coba lagi.", "error")

    return render_template('login.html', next_url=next_url)


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Anda telah berhasil logout.", "info")
    return redirect(url_for('login'))


# ==================== Dashboard ====================

@app.route('/')
@login_required
def dashboard():
    model_pipeline, model_meta = get_active_model()
    active_info = registry.get_active_info()

    stats = {}
    contract_labels, contract_churn_rates = [], []
    payment_labels, payment_churn_rates = [], []
    churn_counts = {0: 0, 1: 0}

    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        stats['total_customers'] = len(df)
        stats['churn_rate'] = round((df['churn'].mean() * 100), 2)
        stats['avg_satisfaction'] = round(df['satisfaction_score'].mean(), 2)
        stats['avg_monthly_spend'] = round(df['monthly_spend_idr'].mean(), 1)
        stats['avg_tenure'] = round(df['tenure_months'].mean(), 1)

        contract_agg = df.groupby('contract_type')['churn'].agg(['count', 'mean']).reset_index()
        contract_labels = contract_agg['contract_type'].tolist()
        contract_churn_rates = (contract_agg['mean'] * 100).round(1).tolist()

        payment_agg = df.groupby('payment_method')['churn'].agg(['count', 'mean']).reset_index()
        payment_labels = payment_agg['payment_method'].tolist()
        payment_churn_rates = (payment_agg['mean'] * 100).round(1).tolist()

        churn_counts = df['churn'].value_counts().to_dict()

    return render_template('index.html',
                           stats=stats,
                           contract_labels=contract_labels,
                           contract_churn_rates=contract_churn_rates,
                           payment_labels=payment_labels,
                           payment_churn_rates=payment_churn_rates,
                           churn_counts=churn_counts,
                           meta=model_meta,
                           active_info=active_info)


# ==================== Predict ====================

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    model_pipeline, model_meta = get_active_model()
    active_info = registry.get_active_info()

    prediction_result = None
    input_values = {
        'age': 38, 'income_million_idr': 8.5, 'tenure_months': 24,
        'monthly_spend_idr': 450, 'support_tickets': 2,
        'login_frequency_monthly': 12.0, 'contract_type': 'Monthly',
        'payment_method': 'Credit Card', 'discount_used': 0,
        'satisfaction_score': 3.0
    }

    # Get optimal threshold from active model
    optimal_threshold = 0.5
    if model_meta and 'optimal_threshold' in model_meta:
        optimal_threshold = model_meta['optimal_threshold']
    elif model_meta and 'test_metrics' in model_meta:
        optimal_threshold = model_meta['test_metrics'].get('optimal_threshold', 0.5)

    if request.method == 'POST':
        form_data = request.form
        input_df, processed_details = preprocess_input(form_data)
        input_values = form_data.to_dict()

        if model_pipeline:
            prob_churn = float(model_pipeline.predict_proba(input_df)[:, 1][0])
            pred_class = int(prob_churn >= optimal_threshold)

            if prob_churn >= 0.60:
                risk_level, risk_color = "High", "danger"
                status_text = "Tinggi (Potensi Kuat Churn)"
            elif prob_churn >= 0.40:
                risk_level, risk_color = "Medium", "warning"
                status_text = "Sedang (Waspada Churn)"
            else:
                risk_level, risk_color = "Low", "success"
                status_text = "Rendah (Pelanggan Loyal)"

            recommendations = []
            if float(form_data.get('satisfaction_score', 5)) <= 2.5:
                recommendations.append("Prioritas Tinggi: Kepuasan kritis. Hubungi via telepon oleh Customer Care.")
            elif float(form_data.get('satisfaction_score', 5)) <= 3.5:
                recommendations.append("Kirimkan survei feedback layanan dan tawarkan bantuan proaktif.")
            if int(form_data.get('support_tickets', 0)) >= 3:
                recommendations.append("Eskalasi tiket bantuan ke tim Priority Support.")
            if form_data.get('contract_type') == 'Monthly':
                recommendations.append("Tawarkan insentif diskon 20% untuk beralih ke paket Annual.")
            if int(form_data.get('discount_used', 0)) == 0:
                recommendations.append("Berikan reward diskon loyalitas pada tagihan berikutnya.")
            if not recommendations:
                recommendations.append("Pertahankan standar pelayanan dan kirimkan buletin pembaruan fitur bulanan.")

            prediction_result = {
                'churn_probability': round(prob_churn * 100, 1),
                'pred_class': pred_class,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'status_text': status_text,
                'recommendations': recommendations,
                'features': processed_details,
                'threshold_used': round(optimal_threshold, 4)
            }
        else:
            flash("Belum ada model aktif. Silakan training model terlebih dahulu.", "warning")

    return render_template('predict.html',
                           result=prediction_result,
                           inputs=input_values,
                           meta=model_meta,
                           active_info=active_info)


# ==================== Batch ====================

@app.route('/batch')
@login_required
def batch_view():
    model_pipeline, model_meta = get_active_model()
    active_info = registry.get_active_info()

    # Get optimal threshold
    optimal_threshold = 0.5
    if model_meta and 'test_metrics' in model_meta:
        optimal_threshold = model_meta['test_metrics'].get('optimal_threshold', 0.5)
    elif model_meta and 'optimal_threshold' in model_meta:
        optimal_threshold = model_meta['optimal_threshold']

    data_sample = []
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        sample_df = df.head(25).copy()

        if model_pipeline:
            sample_df = apply_batch_features(sample_df)
            features_df = sample_df.drop(columns=['customer_id', 'churn'], errors='ignore')
            probs = model_pipeline.predict_proba(features_df)[:, 1]
            sample_df['predicted_churn_prob'] = (probs * 100).round(1)
            sample_df['predicted_churn'] = (probs >= optimal_threshold).astype(int)

        data_sample = sample_df.to_dict(orient='records')

    return render_template('batch.html', data=data_sample, active_info=active_info)


# ==================== Evaluation ====================

@app.route('/evaluation')
@login_required
def evaluation():
    model_pipeline, model_meta = get_active_model()
    active_info = registry.get_active_info()

    if not model_meta:
        flash("Belum ada model aktif. Silakan training model terlebih dahulu.", "warning")
        return render_template('evaluation.html', meta={}, roc_data=[], pr_data=[],
                               feat_labels=[], feat_values=[], active_info=None)

    roc_data = model_meta.get('roc_curve', [])
    pr_data = model_meta.get('pr_curve', [])

    top_features = model_meta.get('feature_importance', [])[:8]
    feat_labels = [f['feature'] for f in top_features]
    feat_values = [f['percentage'] for f in top_features]

    return render_template('evaluation.html',
                           meta=model_meta,
                           roc_data=roc_data,
                           pr_data=pr_data,
                           feat_labels=feat_labels,
                           feat_values=feat_values,
                           active_info=active_info)


# ==================== Model Management ====================

@app.route('/models')
@login_required
def models_page():
    df_models = registry.get_all()
    models_list = df_models.to_dict(orient='records') if not df_models.empty else []
    return render_template('models.html', models=models_list)


@app.route('/models/activate/<versi>', methods=['POST'])
@login_required
def activate_model(versi):
    try:
        registry.activate(versi)
        flash(f"Model {versi} berhasil diaktifkan!", "success")
    except Exception as e:
        flash(f"Gagal mengaktifkan model: {str(e)}", "error")
    return redirect(url_for('models_page'))


@app.route('/models/delete/<versi>', methods=['POST'])
@login_required
def delete_model(versi):
    try:
        registry.delete(versi)
        flash(f"Model {versi} berhasil dihapus.", "info")
    except ValueError as e:
        flash(str(e), "error")
    except Exception as e:
        flash(f"Gagal menghapus model: {str(e)}", "error")
    return redirect(url_for('models_page'))


# ==================== Retrain ====================

@app.route('/retrain', methods=['GET', 'POST'])
@login_required
def retrain():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash("File dataset tidak ditemukan. Silakan upload file.", "error")
            return redirect(url_for('retrain'))

        file = request.files['dataset']
        if file.filename == '':
            flash("Nama file kosong. Silakan pilih file.", "error")
            return redirect(url_for('retrain'))

        if not allowed_file(file.filename):
            flash("Format file tidak didukung. Gunakan .csv, .xlsx, atau .xls", "error")
            return redirect(url_for('retrain'))

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_DIR, saved_filename)
        file.save(filepath)

        # Read file
        try:
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'csv':
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            flash(f"Gagal membaca file: {str(e)}", "error")
            return redirect(url_for('retrain'))

        # Validate columns
        missing_cols = trainer.validate_columns(df)
        if missing_cols:
            flash(f"Kolom wajib tidak ditemukan: {', '.join(missing_cols)}", "error")
            return redirect(url_for('retrain'))

        # Run training
        catatan = request.form.get('catatan', '').strip()
        if not catatan:
            catatan = f"Retrain dengan file {filename}"

        try:
            pipeline, metrics, best_params = trainer.train(
                df, catatan=catatan, dataset_filename=filename
            )

            versi = registry.save(
                pipeline, metrics, best_params,
                catatan=catatan, dataset_filename=filename
            )

            # Auto-activate new model
            registry.activate(versi)

            # Also save as metadata JSON for evaluation page
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

            flash(
                f"Model {versi} berhasil dilatih dan diaktifkan! "
                f"ROC-AUC: {metrics['test_roc_auc']:.4f}, "
                f"Recall: {metrics['test_recall']*100:.1f}%, "
                f"Status: {metrics['model_status']}",
                "success"
            )
            return redirect(url_for('models_page'))

        except Exception as e:
            flash(f"Gagal melatih model: {str(e)}", "error")
            import traceback
            traceback.print_exc()
            return redirect(url_for('retrain'))

    return render_template('retrain.html', preview=None)


@app.route('/model-info')
def model_info_redirect():
    return redirect(url_for('evaluation'))


@app.route('/health')
def health_check():
    """Enterprise healthcheck probe endpoint."""
    pipeline, meta = get_active_model()
    return jsonify({
        "status": "healthy" if pipeline is not None else "degraded",
        "service": "ChurnGuard AI Enterprise",
        "model_loaded": pipeline is not None,
        "active_model": meta.get("model_name", "Unknown") if meta else "None",
        "environment": Config.FLASK_ENV,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json':
        return jsonify({"error": {"code": "NOT_FOUND", "message": "Endpoint not found"}}), 404
    return render_template('base.html', title="404 Not Found"), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/') or request.headers.get('Accept') == 'application/json':
        return jsonify({"error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred"}}), 500
    return render_template('base.html', title="500 Server Error"), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ChurnGuard AI — Enterprise Customer Churn Dashboard")
    print("=" * 60)
    print(f"URL: http://{Config.HOST}:{Config.PORT}")
    print(f"Login: {Config.ADMIN_USERNAME} / {Config.ADMIN_PASSWORD}")
    print(f"Database: {Config.DATABASE_PATH}")
    print("=" * 60)
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)

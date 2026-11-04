# 📊 Customer Churn Prediction System (Metodologi CRISP-DM)

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![CI Tests](https://img.shields.io/badge/Tests-12%2F12%20Passing-brightgreen.svg?logo=github-actions&logoColor=white)](https://github.com/)
[![Security](https://img.shields.io/badge/Security-OWASP%20Hardened-blueviolet.svg)](https://owasp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Sistem Deteksi Retensi & Prediksi Churn Pelanggan Berbasis CRISP-DM Terpadu dengan SQLite Model Registry & Dashboard Flask.**

**Author:** Sofia Azzahra Alamsyah  
**Domain:** Customer Analytics, Subscription Retention, & Business Decision Support  

---

## 🔐 Akun Login Admin (Data Seed)

Untuk mengakses fitur prediksi pelanggan dan data batch pada Web Dashboard, gunakan kredensial berikut:

| Akun | Username | Password | Role |
| :--- | :--- | :--- | :--- |
| **Admin Utama** | `admin` | `admin123` | Administrator (Sofia Azzahra) |
| **Analyst Team** | `analyst` | `analyst123` | Data Analyst |

> *Fitur*: Pada halaman login tersedia fitur **Buka/Tutup Password (Eye Toggle)** dan tombol **"Isi Otomatis"** untuk mempermudah pengujian saat presentasi atau rekrutmen.

---

## 🏗️ Struktur Proyek

```text
Customer churn-portofolio/
│
├── customer_churn_crisp_dm.ipynb   # Jupyter Notebook lengkap 6 fase CRISP-DM
├── customer_churn_crisp_dm.py      # Standalone training & pipeline script (Single Model)
├── app.py                          # Aplikasi Web Dashboard berbasis Flask
│
├── Raw dataset/
│   ├── ml_portfolio_customer_churn_data_dictionary.csv  # Kamus data
│   └── ml_portfolio_customer_churn_dataset.csv          # Dataset utama (1,200 baris)
│
├── static/
│   └── favicon.svg                 # Logo & Favicon web
│
├── templates/                      # Template UI Dashboard Flask (Bootstrap 5 & Chart.js)
│   ├── base.html                   # Layout dasar, navbar, & status login
│   ├── login.html                  # Halaman login admin (dengan eye toggle password)
│   ├── index.html                  # Executive Dashboard & grafik analitis
│   ├── predict.html                # Form simulasi & prediksi churn real-time
│   ├── batch.html                  # Tabel data pelanggan & batch scoring
│   └── evaluation.html             # Halaman Evaluasi Dinamis (grafik, ROC, & matriks)
│
├── models/                         # Model Machine Learning & Metadata
│   ├── best_churn_model.pkl        # Pipeline Random Forest model tersimpan (Joblib)
│   └── model_metadata.json         # Ringkasan dinamis metrik evaluasi & kurva ROC
│
└── reports/
    └── plots/                      # Hasil visualisasi grafik EDA & Evaluasi
        ├── churn_distribution.png
        ├── correlation_heatmap.png
        ├── roc_curves_tuned.png
        ├── confusion_matrix_tuned.png
        └── feature_importances_tuned.png
```

---

## 🔄 Alur Metodologi CRISP-DM

### 1. Phase 1: Business Understanding
* **Masalah Bisnis**: Mempertahankan pelanggan lama (*Customer Retention*) jauh lebih hemat biaya dibandingkan mencari pelanggan baru (*Customer Acquisition Cost* / CAC).
* **Tujuan**: Memprediksi potensi churn pelanggan sedini mungkin agar tim operasional dapat memberikan penawaran retensi yang tepat sasaran.
* **Metrik Utama**: Mengoptimalkan **ROC-AUC** dan **Recall** untuk meminimalkan *False Negative* (pelanggan yang churn namun gagal terdeteksi).

---

### 2. Phase 2: Data Understanding
* **Dimensi**: 1,200 baris data pelanggan dengan 12 fitur mentah.
* **Tingkat Churn**: **17.58%** (211 churn vs 989 aktif) &rarr; *Class Imbalance*.
* **Statistik & EDA**: Analisis deskriptif, matriks korelasi, dan visualisasi distribusi target.

---

### 3. Phase 3: Data Preparation (Analisis Mendalam & Feature Engineering)
* **Pemeriksaan & Imputasi Missing Values**:
  * Ditemukan data kosong pada kolom `income_million_idr` (2.0%), `monthly_spend_idr` (2.0%), dan `satisfaction_score` (2.0%).
  * **Strategi**: Menggunakan **Median Imputation** untuk variabel numerik karena median lebih tahan (*robust*) terhadap skewness dan nilai ekstrim dibandingkan mean.
* **Deteksi Outlier**: Dianalisis menggunakan metode **Interquartile Range (IQR)** pada fitur pengeluaran dan login.
* **Encoding**:
  * `contract_type`: **Ordinal Mapping** (`Monthly` = 1, `Quarterly` = 3, `Annual` = 12 bulan) yang merepresentasikan durasi komitmen kontrak secara numerik.
  * `payment_method`: **One-Hot Encoding** (`drop='first'`).
  * `tenure_segment`: **One-Hot Encoding** (`drop='first'`).
* **Rekayasa Fitur Baru (*Feature Engineering*)**:
  1. `support_tickets_per_tenure`: Rata-rata komplain per bulan masa langganan (korelasi positif terkuat: +0.081).
  2. `is_low_satisfaction`: Indikator biner untuk skor kepuasan pelanggan kritis (&le; 2.5).
  3. `spend_per_login`: Rasio pengeluaran bulanan per sesi login aktif.
  4. `support_to_login_ratio`: Rasio intensitas keluhan dibanding frekuensi login.
  5. `spend_to_income_ratio`: Rasio beban pengeluaran terhadap pendapatan bulanan.
  6. `tenure_segment`: Segmentasi pelanggan (`New`, `Medium`, `Loyal`).
* **Feature Scaling & Data Splitting**: `StandardScaler()` pada fitur numerik dan Stratified Train-Test Split (80% Train, 20% Test).

---

### 4. Phase 4: Modeling (Single Model: Random Forest Classifier)
Model yang digunakan adalah **Random Forest Classifier** yang dioptimalkan menggunakan **5-Fold Stratified Cross Validation** dan **GridSearchCV** dengan pembobotan kelas `class_weight='balanced'`:
* `n_estimators`: 50
* `max_depth`: 5
* `min_samples_split`: 2
* `min_samples_leaf`: 1

---

### 5. Phase 5: Evaluation (Hasil Dinamis pada Data Uji)
Evaluasi dihitung secara dinamis pada 20% data uji (240 data pelanggan):

| Metrik Evaluasi | Nilai Hasil Uji | Keterangan Bisnis |
| :--- | :---: | :--- |
| **ROC-AUC Score** | **0.5437** | Daya pembeda probabilitas model di berbagai ambang batas |
| **Akurasi Total** | **75.0%** | Tingkat ketepatan tebakan secara global |
| **Precision (Churn)** | **17.86%** | Tingkat akurasi saat memprediksi pelanggan churn |
| **Recall (Churn)** | **11.90%** | Cakupan deteksi pelanggan yang sebenarnya churn |
| **F1-Score (Churn)** | **0.1429** | Keseimbangan harmonis antara Precision dan Recall |
| **Brier Score (Loss)** | **0.2027** | Keakuratan kalibrasi estimasi probabilitas model (mendekati 0) |

* **Confusion Matrix (240 Sampel)**:
  * **True Negative (TN)**: 175 (Pelanggan aktif diprediksi benar tetap aktif)
  * **False Positive (FP)**: 23 (*False Alarm* - pelanggan aktif dikira mau churn)
  * **False Negative (FN)**: 37 (Pelanggan churn yang lolos tidak terdeteksi)
  * **True Positive (TP)**: 5 (Pelanggan churn yang berhasil terdeteksi)

---

### 6. Phase 6: Deployment (Web Dashboard Flask)
Model terbaik diekspor menjadi `models/best_churn_model.pkl` dan metadata dinamis ke `models/model_metadata.json`, terintegrasi ke dalam Web Dashboard Flask modern:
* **Fitur Login Admin**: Dilengkapi buka/tutup password dan autofill seed.
* **Dashboard & Visualisasi**: KPI cards dan grafik interaktif Chart.js.
* **Live Single & Batch Prediction**: Perhitungan instan fitur rekayasa dan rekomendasi retensi bisnis.
* **Halaman Evaluasi Dinamis**: Memuat data kurva ROC, Confusion Matrix, dan Feature Importance secara real-time dari metadata JSON.

---

## 🚀 Panduan Menjalankan Project

### 1. Launcher Terpadu Satu Perintah (`run.py`)

Jalankan sistem secara instan menggunakan launcher otomatis (auto-init database & model registry):

```bash
# Menjalankan Web Dashboard Flask & API
python run.py

# Menjalankan seluruh test suite otomatis (12 pengujian)
python run.py --test

# Melatih ulang model machine learning
python run.py --retrain

# Menjalankan di custom port (contoh: 5005)
python run.py --port 5005
```

Buka browser di: **`http://127.0.0.1:5000`** *(Gunakan username `admin` dan password `admin123`)*.

---

### 2. Deployment Kontainer Berbasis Docker

```bash
# Build dan jalankan dengan Docker Compose
docker compose up -d --build

# Periksa status container & healthcheck
docker compose ps
docker compose logs -f
```

---

### 3. Pengujian Otomatis (`pytest`)

```bash
pytest tests/ -v
```

```text
tests/test_churn_app.py::test_health_check_endpoint PASSED               [  8%]
tests/test_churn_app.py::test_security_headers_and_request_id PASSED     [ 16%]
tests/test_churn_app.py::test_login_page_renders PASSED                  [ 25%]
tests/test_churn_app.py::test_unauthenticated_access_redirects_to_login PASSED [ 33%]
tests/test_churn_app.py::test_admin_successful_login PASSED             [ 41%]
tests/test_churn_app.py::test_admin_invalid_login PASSED                [ 50%]
tests/test_churn_app.py::test_authenticated_dashboard_access PASSED     [ 58%]
tests/test_churn_app.py::test_single_prediction_form_get PASSED         [ 66%]
tests/test_churn_app.py::test_single_prediction_post PASSED             [ 75%]
tests/test_churn_app.py::test_evaluation_page_displays_metrics PASSED  [ 83%]
tests/test_churn_app.py::test_models_registry_page PASSED              [ 91%]
tests/test_churn_app.py::test_data_cleaning_service PASSED              [100%]

======================== 12 passed in 3.02s ========================
```

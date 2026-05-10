<p align="center">
  <img src="https://github.com/Viraj-Kumar-Sahu/Car-Price-Prediction/blob/main/Banner%20github.png" width="100%">
</p>

<h1 align="center">🚗 Used Car Price Prediction</h1>

<p align="center">
  Machine Learning Project for Accurate Price Estimation
</p>

---

## 📌 Project Overview

This project predicts used-car selling prices and now includes a reproducible, leakage-safe training pipeline with:

- train-only preprocessing (no leakage)
- repeated K-fold cross-validation + untouched holdout evaluation
- business metrics (`R²`, `MAE`, `RMSE`, `MAPE`, `SMAPE`, `MedAE`)
- segment-wise error analysis (`fuel`, `transmission`, `owner`, `price bucket`)
- time-aware split evaluation
- systematic hyperparameter tuning for top models
- optional native categorical benchmarking (`CatBoost`, `LightGBM`)
- overfitting diagnostics + learning-curve summaries
- permutation importance + optional SHAP importance
- model artifact saving and CLI inference

---

## 📊 Dataset

The project uses the **CarDekho Dataset** (`cardekho.csv`) with these core columns:

- Numerical: `year`, `selling_price`, `km_driven`, `mileage(km/ltr/kg)`, `engine`, `max_power`, `seats`
- Categorical/Text: `name`, `fuel`, `seller_type`, `transmission`, `owner`

---

## 🛠️ Tech Stack

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- XGBoost, CatBoost, LightGBM
- Matplotlib, Seaborn (not required for core CLI)
- Joblib

---

## 🧱 Project Structure

- `src/car_price_prediction/`
  - `config.py`
  - `features.py`
  - `metrics.py`
  - `training.py`
  - `predictor.py`
- `scripts/train.py`
- `scripts/predict.py`
- `Car_Price_Prediction.ipynb` (storytelling/EDA notebook)

---

## 🚀 Reproducible Run (local, no Colab dependency)

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train and evaluate

```bash
PYTHONPATH=src python scripts/train.py \
  --data-path ./cardekho.csv \
  --output-dir ./outputs \
  --artifact-path ./outputs/model.joblib
```

3. Run inference

```bash
PYTHONPATH=src python scripts/predict.py \
  --artifact-path ./outputs/model.joblib \
  --input-csv ./cardekho.csv \
  --output-csv ./outputs/predictions.csv
```

---

## 📁 Outputs Produced by Training

Inside `outputs/`:

- `cv_summary.csv`
- `holdout_summary.csv`
- `segment_error_analysis.csv`
- `time_aware_evaluation.json`
- `overfitting_diagnostics.csv`
- `learning_curve_summary.csv`
- `permutation_importance_top30.csv`
- `shap_importance_top30.csv` (if SHAP runs)
- `summary.json`
- `model.joblib`

---

## ✅ Notes

- The notebook remains available for visualization/storytelling.
- The production workflow should use the modular `src/` + `scripts/` pipeline.

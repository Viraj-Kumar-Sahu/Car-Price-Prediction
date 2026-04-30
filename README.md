<p align="center">
  <img src="https://github.com/Viraj-Kumar-Sahu/Car-Price-Prediction/blob/main/Banner%20github.png" width="100%">
</p>

<h1 align="center">🚗 Used Car Price Prediction</h1>

<p align="center">
  Machine Learning Project for Accurate Price Estimation
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/ML-Regression-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge">
</p>

---

## 📌 Project Overview
> Predicting used car prices using machine learning models

This project aims to predict the selling price of used cars using various machine learning regression algorithms. By analyzing features such as year of manufacture, distance driven, fuel type, and engine specifications, we identify the key drivers of vehicle valuation.

---

## 📊 Dataset
> CarDekho dataset with 8,129 entries and 12 features

The project uses the **CarDekho Dataset** from **Kaggle**, which contains 8,129 entries with 12 initial features:

- **🔢 Numerical:** Year, Selling Price, KM Driven, Mileage, Engine, Max Power, Seats  
- **🔤 Categorical:** Fuel Type, Seller Type, Transmission, Owner  

---

## 🛠️ Tech Stack
> Tools and technologies used in this project

- 🐍 **Python**  
- 📊 **Pandas, NumPy**  
- 🤖 **Scikit-Learn, XGBoost**  
- 📈 **Matplotlib, Seaborn**  

---

## 🚀 Project Pipeline
> Step-by-step workflow of the project

1. 🧹 **Data Preprocessing:** Handled missing values via median imputation and performed One-Hot Encoding  
2. 📊 **EDA:** Visualized price distributions and feature correlations  
3. ⚖️ **Feature Scaling:** Applied `StandardScaler` for KNN and SVR  
4. 🤖 **Model Benchmarking:** Evaluated 11 regression models  
5. 🎯 **Hyperparameter Tuning:** Optimized Decision Tree depth  

---

## 📈 Model Performance Summary

| Model | R2 Score | MAE | RMSE | Adjusted R2 |
|------|--------|--------|--------|--------|
| 🏆 **Random Forest** | **0.9686** | ₹70,457 | ₹143,515 | 0.9683 |
| Gradient Boosting | 0.9614 | ₹92,388 | ₹159,136 | 0.9610 |
| Decision Tree (Tuned) | 0.9605 | ₹84,325 | ₹160,940 | 0.9601 |
| XGBoost | 0.9605 | ₹71,378 | ₹160,971 | 0.9601 |
| Decision Tree (Base) | 0.9566 | ₹82,177 | ₹168,655 | 0.9562 |
| KNN Regressor | 0.9455 | ₹96,817 | ₹189,086 | 0.9449 |
| SVR (RBF) | 0.9284 | ₹98,895 | ₹216,632 | 0.9277 |
| Linear Regression | 0.6979 | ₹268,887 | ₹445,015 | 0.6949 |
| Lasso Regression | 0.6979 | ₹268,886 | ₹445,020 | 0.6949 |
| Ridge Regression | 0.6964 | ₹269,194 | ₹446,104 | 0.6934 |
| AdaBoost | 0.6150 | ₹470,720 | ₹502,359 | 0.6112 |

---

## 🏆 Key Insights

✨ **Best Model:** Random Forest (96.86% R²)  
⚡ **Top Features:** `max_power`, `engine`, `year`  
📉 **Scaling Impact:** Improved KNN & SVR performance  

---

## 📬 Contact

👨‍💻 **Team Members**

- Viraj Kumar Sahu  
- Deepdas Somani  
- Ayan Bhandari  

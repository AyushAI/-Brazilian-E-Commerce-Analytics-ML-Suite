# 🇧🇷 Brazilian E-Commerce Analytics & ML Suite (Olist Dataset)

An end-to-end Data Analytics & Machine Learning project built on the Brazilian E-Commerce Public Dataset by Olist.  

This project covers:
- Extensive Exploratory Data Analysis (EDA)
- Statistical Hypothesis Testing (ANOVA, Chi-Square)
- Machine Learning (Classification & Regression)
- Clustering
- NLP Sentiment Analysis
- Time Series Forecasting (ARIMA, SARIMA, Prophet)
- Streamlit Deployment (Interactive Web App)

---

## 📌 Project Overview

This project aims to extract actionable business insights from Olist’s Brazilian e-commerce dataset and build predictive systems for:

- 🎯 Customer Churn Prediction (Classification)
- 💰 Payment Value Prediction (Regression)
- 🧠 Customer Review Sentiment Analysis (NLP)
- 📈 Sales Forecasting (Time Series)
- 👥 Customer Segmentation (Clustering)

The solution follows a structured end-to-end ML pipeline with proper statistical validation and model comparison.

---

## 📂 Dataset

**Source:** Olist Brazilian E-Commerce Dataset  
Contains information about:
- Customers
- Orders
- Payments
- Reviews
- Products
- Sellers
- Geolocation

---

# 🔎 1. Exploratory Data Analysis (EDA)

Performed extensive data exploration including:

- Missing value analysis
- Outlier detection
- Distribution analysis
- Correlation heatmaps
- Category-level performance analysis
- Time-based order trends
- Revenue analysis by state and product category

---

# 📊 2. Statistical Hypothesis Testing

To validate business assumptions, the following tests were conducted:

### ✔ ANOVA
- Compared payment values across multiple product categories
- Tested revenue differences across regions

### ✔ Chi-Square Test
- Relationship between:
  - Payment type & customer churn
  - Review score & churn
  - Delivery delay & customer satisfaction

These tests provided statistical backing to business insights.

---

# 🤖 3. Machine Learning Models

## 🎯 A. Classification – Customer Churn Prediction

### Models Implemented:
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Ridge / Lasso (where applicable)
- Additional ensemble comparisons

### Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

📌 Best model selected based on cross-validation and generalization performance.

---

## 💰 B. Regression – Payment Value Prediction

### Models Implemented:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

### Evaluation Metrics:
- MAE
- RMSE
- R² Score

Feature importance visualization included for interpretability.

---

# 👥 4. Clustering Module

### Models:
- KMeans
- DBSCAN

### Evaluation:
- Silhouette Score
- Cluster Visualization
- Business Interpretation of Segments

Used to identify:
- High-value customers
- Price-sensitive buyers
- Repeat vs one-time customers

---

# 🧠 5. NLP Module – Review Sentiment Analysis

## Text Preprocessing:
- Lowercasing
- Punctuation removal
- Stopword removal
- Tokenization

## Text Representation:
- TF-IDF
- Optional: Word2Vec

## Tasks:
- Sentiment Classification
- Top Keyword Extraction
- Basic Named Entity Recognition

### Evaluation:
- Accuracy
- Precision / Recall
- F1 Score
- Confusion Matrix

### Outputs:
- Sentiment distribution chart
- Top keywords visualization
- Extracted entity examples

---

# 📈 6. Time Series Forecasting Module

Forecasted future sales trends.

### Models:
- ARIMA
- SARIMA
- Prophet

### Steps:
- Stationarity check (ADF Test)
- Differencing
- Seasonal decomposition
- Model comparison
- 6–12 month forecasting

### Evaluation:
- MAE
- RMSE

Compared models to select best-performing forecasting approach.

---

# ⚙ 7. End-to-End ML Pipeline

Implemented using:

- scikit-learn Pipelines
- Cross-validation
- Feature Engineering
- Train/Test Split
- Model Comparison
- Final Model Selection

Ensured reproducibility and modularity.

---

# 🌐 8. Streamlit Web Application

Built an interactive Streamlit app for:

- Customer churn prediction
- Payment value prediction
- Sentiment classification
- Forecast visualization

### Features:
- User input form
- Real-time prediction
- Model loading via serialized objects
- Interactive charts

To run locally:

```bash
streamlit run app.py

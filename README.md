# Income Classification Project

## 📌 Overview

This project aims to predict whether an individual earns more than $50,000 per year based on demographic and employment-related features. Using a dataset of approximately 48,000 individuals, the project implements the full CRISP-DM methodology — from data understanding to model evaluation — to build robust and interpretable classification models.

This task supports applications in public policy, socio-economic research, and marketing by leveraging machine learning to extract insights from real-world data.

---

## 📂 Dataset

- **Source**: Provided via course platform (INF5082)
- **Size**: ~48,000 instances
- **Target**: Binary classification  
  - `>50K`: earns more than \$50,000/year  
  - `<=50K`: earns \$50,000/year or less
- **Features**: 14 input variables  
  - **Numerical**: `age`, `fnlwgt`, `capital-gain`, `capital-loss`, `hours-per-week`  
  - **Categorical (Nominal)**: `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
  - **Categorical (Ordinal)**: `education` (encoded as `education-num`)  

Missing values are marked with `?`.

---

## 🧭 Project Objectives

1. **Data Cleaning & Preprocessing**
   - Impute missing values using statistical and model-based techniques
   - Encode categorical features appropriately
   - Normalize numerical data

2. **Exploratory Data Analysis (EDA)**
   - Analyze distributions, central tendencies, and dispersion
   - Visualize data using histograms, boxplots, count plots, and heatmaps
   - Detect outliers and correlations with the target

3. **Class Balancing**
   - Address class imbalance using:
     - SMOTE
     - Random over/under-sampling
     - Class weight adjustments in models

4. **Model Development**
   - Train and tune 7 classification algorithms:
     - Logistic Regression  
     - Decision Tree  
     - Random Forest  
     - K-Nearest Neighbors (KNN)  
     - Support Vector Machine (SVM)  
     - Naive Bayes  
     - Gradient Boosting  

5. **Model Evaluation**
   - Use 70/30 train-test split
   - Perform k-fold cross-validation (k = 5, 7, 10)
   - Evaluate with metrics:
     - Accuracy, Precision, Recall, F1-score, AUC-ROC
   - Compare performance across models and validation strategies
   - Analyze overfitting by comparing train/test and cross-validation results

6. **Recommendations**
   - Identify the best-performing models for deployment
   - Propose improvements for future iterations (feature engineering, hyperparameter tuning, etc.)

---

## 🧪 Technologies Used

- **Python**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — visualization
  - `scikit-learn` — machine learning and preprocessing
  - `imblearn` — class balancing (e.g., SMOTE)
  - `msnow` — missing value analysis

---

## 💬 Citation
> INF5082 – TP1: ANALYSE EXPLORATOIRE ET MODÉLISATION DE DONNÉES – Income Dataset Component, Université du Québec à Montréal, Summer 2025.

---

## 👤 Authors

**Names**: Églantine Clervil and Yasmine Naas
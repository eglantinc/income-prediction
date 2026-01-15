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

## 📊 Performance on full income dataset

**Cross-validation (k=5)**

| Model               | Accuracy        | Precision       | Recall          | F1              | AUC ROC         |
|---------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Logistic Regression | 0.8048 ± 0.0049 | 0.5623 ± 0.0072 | 0.8350 ± 0.0083 | 0.6720 ± 0.0072 | 0.9018 ± 0.0044 |
| Decision Tree       | 0.8015 ± 0.0039 | 0.5777 ± 0.0068 | 0.6359 ± 0.0147 | 0.6054 ± 0.0097 | 0.7449 ± 0.0070 |
| Random Forest       | 0.8347 ± 0.0007 | 0.6457 ± 0.0038 | 0.6863 ± 0.0136 | 0.6653 ± 0.0047 | 0.8909 ± 0.0022 |
| KNN                 | 0.7875 ± 0.0028 | 0.5392 ± 0.0044 | 0.7744 ± 0.0051 | 0.6357 ± 0.0033 | 0.8484 ± 0.0030 |
| Naive Bayes         | 0.8189 ± 0.0045 | 0.6412 ± 0.0129 | 0.5535 ± 0.0111 | 0.5940 ± 0.0096 | 0.8704 ± 0.0046 |
| Gradient Boosting   | 0.8216 ± 0.0044 | 0.5895 ± 0.0078 | 0.8394 ± 0.0066 | 0.6926 ± 0.0056 | 0.9146 ± 0.0017 |
| SVM                 | 0.8017 ± 0.0025 | 0.5569 ± 0.0038 | 0.8414 ± 0.0040 | 0.6702 ± 0.0033 | 0.9011 ± 0.0017 |

**Cross-validation (k=7)**

| Model               | Accuracy        | Precision       | Recall          | F1              | AUC ROC         |
| ------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Logistic Regression | 0.8046 ± 0.0059 | 0.5618 ± 0.0088 | 0.8365 ± 0.0095 | 0.6722 ± 0.0088 | 0.9018 ± 0.0053 |
| Decision Tree       | 0.8013 ± 0.0042 | 0.5769 ± 0.0080 | 0.6385 ± 0.0135 | 0.6061 ± 0.0093 | 0.7457 ± 0.0070 |
| Random Forest       | 0.8343 ± 0.0051 | 0.6437 ± 0.0130 | 0.6907 ± 0.0111 | 0.6663 ± 0.0080 | 0.8910 ± 0.0037 |
| KNN                 | 0.7877 ± 0.0044 | 0.5397 ± 0.0071 | 0.7740 ± 0.0068 | 0.6359 ± 0.0044 | 0.8478 ± 0.0039 |
| Naive Bayes         | 0.8191 ± 0.0054 | 0.6417 ± 0.0141 | 0.5540 ± 0.0124 | 0.5946 ± 0.0116 | 0.8704 ± 0.0050 |
| Gradient Boosting   | 0.8213 ± 0.0065 | 0.5888 ± 0.0101 | 0.8420 ± 0.0119 | 0.6930 ± 0.0101 | 0.9147 ± 0.0051 |
| SVM                 | 0.8012 ± 0.0054 | 0.5561 ± 0.0077 | 0.8416 ± 0.0084 | 0.6697 ± 0.0077 | 0.9011 ± 0.0045 |

**Cross-validation (k=10)**

| Model               | Accuracy        | Precision       | Recall          | F1              | AUC ROC         |
| ------------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| Logistic Regression | 0.8048 ± 0.0023 | 0.5621 ± 0.0035 | 0.8349 ± 0.0038 | 0.6719 ± 0.0029 | 0.9018 ± 0.0017 |
| Decision Tree       | 0.8015 ± 0.0074 | 0.5776 ± 0.0160 | 0.6384 ± 0.0121 | 0.6064 ± 0.0118 | 0.7457 ± 0.0077 |
| Random Forest       | 0.8330 ± 0.0052 | 0.6416 ± 0.0128 | 0.6862 ± 0.0119 | 0.6630 ± 0.0091 | 0.8912 ± 0.0047 |
| KNN                 | 0.7880 ± 0.0053 | 0.5402 ± 0.0084 | 0.7725 ± 0.0076 | 0.6358 ± 0.0064 | 0.8486 ± 0.0061 |
| Naive Bayes         | 0.8190 ± 0.0052 | 0.6414 ± 0.0151 | 0.5541 ± 0.0116 | 0.5945 ± 0.0104 | 0.8705 ± 0.0061 |
| Gradient Boosting   | 0.8219 ± 0.0068 | 0.5900 ± 0.0115 | 0.8406 ± 0.0142 | 0.6933 ± 0.0098 | 0.9147 ± 0.0052 |
| SVM                 | 0.8008 ± 0.0065 | 0.5554 ± 0.0096 | 0.8427 ± 0.0089 | 0.6695 ± 0.0089 | 0.9011 ± 0.0052 |

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

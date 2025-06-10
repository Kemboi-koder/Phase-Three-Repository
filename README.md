Credit Card Default Classification Project
📝 Problem Statement
Lifeline Creditors is a company that has faced numerous losses due to credit card customers defaulting on their payments during the last financial year. To cut losses, the company has opted to reduce the quantity of high-risk loans. You are tasked with developing a machine learning algorithm that classifies credit card borrowers to find the most likely to default on payment the next month. This project aims to build a machine learning model to classify whether a customer will default on their credit card payment next month using UCI Loan Data a case study of Taiwan. This prediction can guide risk assessment and help banks minimize financial risk.

Stakeholders:

Credit Risk Managers
Loan Approval Officers
Data Analysts

## 📌 Overview

This project aims to predict whether a customer is likely to default on their credit card payments in the upcoming month using a **Decision Tree Classifier** with **threshold tuning** for performance optimization.

## 📊 Dataset

The dataset used contains anonymized customer data including:
- Credit limit
- Demographics (age, sex, education)
- Monthly payment behavior
- Bill statements

## 🧠 Methodology

1. **Preprocessing**: Encoding categorical variables, handling class imbalance with SMOTE.
2. ![image](https://github.com/user-attachments/assets/e114792f-f7ea-4b8a-8003-8fd338fe0474)

3. 
4. **Modeling**: Fitting a logistic regression model. Train a Decision Tree Classifier.
5. **Threshold Tuning**: F1-score, precision, and recall plotted to find optimal decision threshold.
6. **Evaluation**:
   - ROC AUC Score
   - Confusion Matrix
   - Decision Tree Visualization

## 🔍 Key Results

- **ROC AUC**: 0.71
- **True Positives (Defaults Detected)**: 907
- **Precision vs. Recall trade-offs** analyzed for business alignment.

## 📈 Visualizations

- Decision Tree plot (depth=3)
- 
- Confusion Matrix Heatmap
- ROC Curve
- Threshold Tuning Graph (Precision, Recall, F1)

## ✅ Recommendations

- Deploy as a pre-screening tool in lending workflows.
- Use ensemble models for improved stability.
- Perform fairness testing to ensure ethical AI use.

## ⚠️ Limitations

- Historical data only (no temporal dynamics).
- Risk of overfitting with a single tree.
- Imbalanced classes may still affect model performance.

## 🔧 Tools Used

- Python, Scikit-learn, Matplotlib, Seaborn, Pandas
- Jupyter Notebook

## 📂 Project Structure


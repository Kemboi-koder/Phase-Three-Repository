# Credit Default Prediction Using Decision Tree

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
2. **Modeling**: Trained a Decision Tree Classifier.
3. **Threshold Tuning**: F1-score, precision, and recall plotted to find optimal decision threshold.
4. **Evaluation**:
   - ROC AUC Score
   - Confusion Matrix
   - Decision Tree Visualization

## 🔍 Key Results

- **ROC AUC**: 0.71
- **True Positives (Defaults Detected)**: 907
- **Precision vs. Recall trade-offs** analyzed for business alignment.

## 📈 Visualizations

- Decision Tree plot (depth=3)
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


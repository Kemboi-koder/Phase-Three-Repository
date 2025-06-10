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

The dataset used contains anonymized customer data, including:
- Credit limit
- Demographics (age, sex, education)
- Monthly payment behavior
- Bill statements

## 🧠 Methodology

1. **Preprocessing**: Encoding categorical variables, handling class imbalance with SMOTE.
2. Convert the categorical columns (SEX, EDUCATION, MARRIAGE) into dummy variables using one-hot encoding, then drop the first category to prevent multicollinearity. This prepares the model for log regression.
3. 
4. ![image](https://github.com/user-attachments/assets/e114792f-f7ea-4b8a-8003-8fd338fe0474)

5. The "defaulted" class seems to be imbalanced. We initialize SMOTE, which will create synthetic data points in the minority class to oversample it. We import SMOTE from imblearn and import the train_test_split sklearn. We then train the model on the synthetic data points.
6. 
7. **Modeling**: Fitting a logistic regression model. Train a Decision Tree Classifier.
8. Now that all our columns are in numeric form and class imbalance has been resolved, we perform a logistic regression. Fit the trained data onto a logistic regression model to predict whether a customer will default. This model serves as a strong and interpretable baseline. We use **sklearn** to perform the regression to get the regression report.
9. Classification Report:

              precision    recall  f1-score   support

           0       0.84      0.58      0.69      4673
           1       0.29      0.61      0.40      1327

    accuracy                           0.59      6000
   macro avg       0.57      0.60      0.54      6000
weighted avg       0.72      0.59      0.62      6000

10. The model is better at detecting defaulters (recall = 61%) than at correctly predicting them (precision = 29%). -Overal recall of the minority class after SMOTE is good but there’s still a high false positive rate
12. 
13. **Threshold Tuning**: F1-score, precision, and recall plotted to find optimal decision threshold.
14. ![image](https://github.com/user-attachments/assets/630d7801-62ab-4266-97d3-ee6fe8e742e8)

15. We use threshold tuning to find a balanced trade-off between recall and precision.
16.    precision    recall  f1-score   support

           0       0.90      0.22      0.35      4673
           1       0.25      0.91      0.39      1327

    accuracy                           0.37      6000
   macro avg       0.57      0.56      0.37      6000
weighted avg       0.75      0.37      0.36      6000

16. **Evaluation**:
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


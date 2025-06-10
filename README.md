# Credit Card Default Classification Project
## 📝 Problem Statement
Lifeline Creditors is a company that has faced numerous losses due to credit card customers defaulting on their payments during the last financial year. To cut losses, the company has opted to reduce the quantity of high-risk loans. You are tasked with developing a machine learning algorithm that classifies credit card borrowers to find the most likely to default on payment the next month. This project aims to build a machine learning model to classify whether a customer will default on their credit card payment next month using UCI Loan Data, a case study of Taiwan. This prediction can guide risk assessment and help banks minimize financial risk.

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
- 
  ## 🔧 Tools Used
- Python, Scikit-learn, Matplotlib, Seaborn, Pandas
- Jupyter Notebook

# 📂 Project Structure

## 🧠 Methodology

## 1.**Preprocessing**: Encoding categorical variables.
2. Convert the categorical columns (SEX, EDUCATION, MARRIAGE) into dummy variables using one-hot encoding, then drop the first category to prevent multicollinearity. This prepares the model for log regression.
 
 ![image](https://github.com/user-attachments/assets/e114792f-f7ea-4b8a-8003-8fd338fe0474)

 The "defaulted" class seems to be imbalanced. We initialize SMOTE, which will create synthetic data points in the minority class to oversample it. We import SMOTE from imblearn and import the train_test_split sklearn. We then train the model on the synthetic data points.
 
## 2. **Modeling**: Fitting a logistic regression model. Train a Decision Tree Classifier.
 Now that all our columns are in numeric form and class imbalance has been resolved, we perform a logistic regression. Fit the trained data onto a logistic regression model to predict whether a customer will default. This model serves as a strong and interpretable baseline. We use **sklearn** to perform the regression to get the regression report.
Classification Report:

              precision    recall  f1-score   support

           0       0.84      0.58      0.69      4673
           1       0.29      0.61      0.40      1327

    accuracy                           0.59      6000
   macro avg       0.57      0.60      0.54      6000
weighted avg       0.72      0.59      0.62      6000

The model is better at detecting defaulters (recall = 61%) than at correctly predicting them (precision = 29%). -Overal recall of the minority class after SMOTE is good but there’s still a high false positive rate

## 3. **Threshold Tuning**
     F1-score, precision, and recall plotted to find optimal decision threshold.
 ![image](https://github.com/user-attachments/assets/630d7801-62ab-4266-97d3-ee6fe8e742e8)

We use threshold tuning to find a balanced trade-off between recall and precision.
    precision    recall  f1-score   support

           0       0.90      0.22      0.35      4673
           1       0.25      0.91      0.39      1327

    accuracy                           0.37      6000
   macro avg       0.57      0.56      0.37      6000
weighted avg       0.75      0.37      0.36      6000

## 4.**📈 Visualizations and Evaluations**:
### ROC AUC Score
   - ![image](https://github.com/user-attachments/assets/fd4465e4-cacf-4cbb-9a83-7674b7c541be)![image](https://github.com/user-attachments/assets/627dfb12-b6ff-4377-bdd7-527aea33641b)

   - The AUC (area under the curve) after threshold optimization is 0.71, which is an improvement from 0.63 in the previous model, meaning that this model is more accurate at a threshold of 0.37 from the previous default threshold of 0.5

  ### Confusion Matrix
  ![image](https://github.com/user-attachments/assets/4b850dfe-c6d3-44f0-9afc-2aabd4089c4c)
  
True Positives (TP) = 907 → Correctly predicted defaults.
True Negatives (TN) = 2896 → Correctly predicted non-defaults.
False Positives (FP) = 1777 → Non-defaults incorrectly predicted as defaults.
False Negatives (FN) = 420 → Defaults missed by the model.
This matrix reflects a balanced trade-off:

-The model captures a large portion of defaults (good recall).
-However, there is still a notable rate of false positives, which could result in wrongly flagged customers.
-Aligns with our goal of risk mitigation, where catching defaults is more critical than some over-warning.

### Decision Tree Visualization
![image](https://github.com/user-attachments/assets/618e5191-5873-4b1e-8d62-66053c7a8f9d)
The tree highlights that recent repayment history (PAY_* variables) and demographic features (like gender or marital status) influence the likelihood of default.
MARRIAGE (≤1.5): The root split suggests marital status is the strongest predictor

PAY Variables: Payment history features dominate the tree structure (PAY_0, PAY_2: Recent payment behavior)

EDUCATION (≤2.5): Educational attainment influences risk. Graduate school (1), University (2), High school(3), Others (4)

BILL_AMT2 & PAY_AMT2: Financial capacity indicators. Bill amounts show spending patterns

## 🔍 Key Results
-The AUC (area under the curve) is 0.71, which is an improvement from 0.63 in the previous model, meaning that this model is more accurate at a threshold of 0.37, from the previous default threshold of 0.5

-The model is better at detecting defaulters (recall = 68%) form the previous which was (recall=61%).

-Precision has inrceased to 34% from 29% in the previous model which means that this new model is better at correctly predicting deafaulters(True Positives).

-Overall recall of the minority class after threshold tuning has increased as well as the precision, meaning the false positive rate has reduced.

-In conclusion the new model is better at detecting and predicting possible defaulters while reducing the number of non-defaulters flagged as compared to the previous model.

## 5: ✅ Recommendations
-Clients with low credit limits and inconsistent past payments should be flagged for review.

-Younger clients and those with high monthly bill statements may require stricter approval conditions.

Clients with lower education levels have a higher loan default rate, suggesting an element of financial literacy in funds management.

Married clients have a better loan repayment rate than single ones. This could possibly be due to the fact that having two contributors in a household eases the burden on financial resources.

-Incorporate this model into your approval system for early risk identification.

## 6:📉 Study Limitations
-The dataset is based on historical data from Taiwan and may not generalize well to other regions.

-Categorical variables like EDUCATION and MARRIAGE were grouped manually and may affect model accuracy.

-The model might discriminate along the lines of categorical data depending on the inherited bias from the train data.

-Logistic Regression may underfit, and Decision Trees may overfit without pruning or ensemble methods.






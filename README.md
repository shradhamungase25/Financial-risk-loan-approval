Financial Risk for Loan Approval Group Project Titan Miners Our project analyzes the Financial Risk for Loan Approval dataset from Kaggle and builds three predictive models to determine the likelihood that a loan application is approved. The goal is to compare different model approaches and evaluate which provides the strongest predictive performance.

Research Questions The project focuses on understanding key drives of loan approval decisions. Specifically, we are aiming to answer:

Which financial varaibles (i.e. income, credit score, debt levels) matter most in predicting approval?
Do personal factors (i.e. education, employment status, or marital status) influence outcomes?
Do loan details (i.e. loan amount, loan purpose) play a significant role in approval likelihood?
Datset Overview The dataset includes borrower financial and demographic information relevant to determining loan risk. Including variables such as:

Income
Employment details
Credit history
Existing debt
Loan amount requested
Previous defaults
The target variable is Loan Status, indicating whether the loan was approved.

Project Workflow

Data Cleaning + Preprocessing:
Handling missing values if applicable
Encoding categorical features
Feature scaline if applicable
Train/test split
Model Approaches:
Logistic Regression: Baseline interpretable model for binary classification
Decision Tree: Non-linear model that captures feature interactions
Random Forest: Ensemble method used to improve accuracy and reduce overfitting
Model Evaluation:
Models were compared using metrics such as: Accuracy, Precision, ROC AUC

# Credit Risk / Loan Approval Prediction

## Project Overview
This project focuses on predicting **loan approval** using historical loan applicant data. The goal is to help banks identify high-risk borrowers and make informed lending decisions. This dataset is a classic example of a **credit risk analysis** problem, suitable for a Credit Risk Analyst role.

The project covers the **complete data analytics workflow**, including:
- Data cleaning and preprocessing
- Feature engineering
- Exploratory Data Analysis (EDA)
- Machine learning model building and evaluation
- Visualization and insights

---

## Dataset
The dataset used contains **614 loan applications** with 13 features:

| Feature               | Description |
|-----------------------|-------------|
| Loan_ID               | Unique loan ID |
| Gender                | Male / Female |
| Married               | Yes / No |
| Dependents            | Number of dependents (0,1,2,3+) |
| Education             | Graduate / Not Graduate |
| Self_Employed         | Yes / No |
| ApplicantIncome       | Applicant's monthly income |
| CoapplicantIncome     | Co-applicant's monthly income |
| LoanAmount            | Loan amount in thousands |
| Loan_Amount_Term      | Term of the loan in months |
| Credit_History        | 1 = meets credit history requirements, 0 = does not |
| Property_Area         | Rural / Semiurban / Urban |
| Loan_Status           | Target variable: Y = Approved, N = Not Approved |

> Note: Missing values were handled, categorical variables encoded, and new features engineered for model training.

---

## Key Steps

### 1. Data Preprocessing
- Imputed missing values:
  - Categorical → Mode
  - Numerical → Median
- Converted `Dependents` from `'3+'` to `3` (numeric)
- Created `TotalIncome = ApplicantIncome + CoapplicantIncome`
- Log-transformed `LoanAmount` and `TotalIncome` to reduce skewness
- Encoded categorical variables using **One-Hot Encoding**
- Mapped `Loan_Status` → 1 (Approved), 0 (Not Approved)

### 2. Exploratory Data Analysis (EDA)
- Checked distribution of loan approvals
- Visualized correlations between numerical features
- Analyzed loan status trends by **Education**, **Property_Area**, and **Credit_History**

### 3. Model Building
Trained multiple machine learning models to predict loan approval:

| Model                  | Purpose |
|------------------------|---------|
| Logistic Regression     | Baseline model for probability prediction |
| Decision Tree           | Non-linear model capturing complex patterns |
| Random Forest           | Ensemble model with feature importance analysis |

**Evaluation Metrics:**
- Accuracy  
- Precision / Recall / F1-score  
- ROC-AUC  

### 4. Insights & Feature Importance
- **Credit_History** is the strongest predictor of loan approval  
- Higher **TotalIncome** increases chances of approval  
- Applicants from **Urban/Semiurban** areas have slightly higher approval rates  

### 5. Visualizations
- Loan Status Distribution  
- Boxplots: TotalIncome vs Loan_Status, LoanAmount vs Loan_Status  
- Correlation heatmap  
- ROC curve for model evaluation  

---

## Technologies & Tools Used
- Python (pandas, numpy, matplotlib, seaborn, scikit-learn)  
- Machine Learning: Logistic Regression, Decision Tree, Random Forest  
- Data Visualization: Seaborn, Matplotlib  

---

# Insurance Fraud Detection using Machine Learning

## Project Overview

This project detects fraudulent insurance claims using Machine Learning.
The system analyzes insurance claim data and predicts whether a claim is **Legitimate** or **Fraudulent**.

The project implements the complete ML workflow:

- Problem Understanding
- Data Preparation
- Exploratory Data Analysis
- Model Building
- Hyperparameter Tuning
- Model Deployment using Flask
- Final Demonstration Dashboard

This project was developed as part of the Skill Wallet Machine Learning Project.

---

## Dataset

Dataset used: insurance_claims.csv

- Records: 1000
- Features: 40
- Target: fraud_reported
- Classes:
  - Y → Fraud
  - N → Legitimate

Dataset is imbalanced, so SMOTE was used.

---

## Project Structure
skillwallet_project/
├── templates/
│ ├── home.html
│ ├── about.html
│ ├── predict.html
│ └── result.html
│
├── static/
│ ├── bg.png
│ └── br.png
│
├── SkillWallet_Final1.ipynb
├── epic6_deployment.py
├── tuned_model.pkl
├── prepared_data.pkl
├── insurance_claims.csv
├── epic7_final_dashboard.png
├── README.md
└── .gitignore


---

## Machine Learning Workflow

### Epic 1 – Problem Understanding

- Loaded dataset
- Checked class imbalance
- Identified target variable
- Visualized fraud distribution

### Epic 2 – Data Preparation

- Missing value handling
- Encoding categorical data
- Feature engineering
- SMOTE balancing
- Train / test split
- Saved prepared data

### Epic 3 – Exploratory Data Analysis

- Distribution plots
- Fraud rate by category
- Correlation heatmap
- Pattern analysis
- Outlier detection

### Epic 4 – Model Building

Models trained:

- Decision Tree
- Random Forest
- KNN
- Logistic Regression
- Naive Bayes
- SVM

Best model selected based on accuracy.

### Epic 5 – Hyperparameter Tuning

Used GridSearchCV to tune model.

Saved model:
tuned_model.pkl


### Epic 6 – Model Deployment (Flask)

Flask web app with pages:

- Home
- About
- Predict
- Result

Run:
python epic6_deployment.py


### Epic 7 – Project Demonstration

Final dashboard includes:

- Confusion Matrix
- ROC Curve
- Feature Importance
- Model Metrics
- Workflow Summary

Saved as:
epic7_final_dashboard.png



---

## Model Performance

Example results:

- Accuracy ≈ 0.74 – 0.75
- Precision moderate
- Recall low (fraud detection is difficult)
- ROC-AUC ≈ ~0.5

Fraud detection is challenging because of class imbalance.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML / CSS / Bootstrap
- Git & GitHub

---

## How to Run the Project

Clone repository
git clone https://github.com/vasudha-star/Skill-Wallet-ML-Project.git


Go to folder
cd Skill-Wallet-ML-Project

Install libraries
pip install pandas numpy scikit-learn flask matplotlib seaborn


Run Flask app
python epic6_deployment.py


---
#Authors
Madugula Vasudha
Pasham Srikar
Joy Soni
Chintha Rohith Sai Akshay

## Notes

This project is for educational purposes and demonstrates the complete Machine Learning pipeline including deployment using Flask.

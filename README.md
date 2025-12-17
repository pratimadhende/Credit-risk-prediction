# 📊 Credit Risk Prediction using Machine Learning

## 📌 Project Overview
Credit risk assessment is a critical task in the banking and financial sector.  
This project focuses on building a *machine learning model* to predict whether a loan applicant is likely to *default or repay* a loan based on their financial and personal attributes.

The goal is to help financial institutions *minimize risk, reduce losses, and make data-driven lending decisions*.

---

## 🎯 Objective
- Predict *loan default risk* using supervised machine learning  
- Analyze key factors influencing credit risk  
- Compare multiple ML models and select the best-performing one  
- Provide interpretable and reproducible results  

---

## 📂 Dataset
- *Source:* Kaggle – Credit Risk Dataset  
- *Link:* https://www.kaggle.com/datasets/laotse/credit-risk-dataset  
- *Description:*  
  The dataset contains applicant information such as income, loan amount, credit history, employment length, and loan purpose.

---

## 🧠 Machine Learning Approach
This is a *binary classification problem* where the target variable indicates:
- 0 → No Default  
- 1 → Default  

### Models Used:
- Logistic Regression  
- Random Forest Classifier  
- Support Vector Machine (SVM)  

The *Random Forest model* achieved the best overall performance.

---

## 🔧 Technologies & Tools
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Joblib  

---

## 🛠 Project Workflow
1. Data Loading & Exploration  
2. Data Cleaning & Missing Value Treatment  
3. Feature Encoding & Scaling  
4. Train-Test Split  
5. Model Training & Comparison  
6. Model Evaluation (Accuracy, ROC-AUC, Confusion Matrix)  
7. Feature Importance Analysis  
8. Model Saving (.pkl format)  

---

## 📈 Model Evaluation Metrics
- Accuracy Score  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC-AUC Curve  

These metrics ensure both *performance and reliability* of the model.

---

## 📊 Visualizations
The project includes the following saved visual outputs:
- Confusion Matrix  
- ROC Curve  
- Feature Importance Plot  

These visualizations help in *model interpretability and decision-making*.

---

## 🚀 Results & Insights

Random Forest outperformed other models based on ROC-AUC score

Credit history, income, and loan amount were the most influential features

The model provides reliable predictions for credit risk assessment



---

## 👤 Author

Pratima Dhende
B.Sc. Computer Science (Final Year)

---

## ⭐ Conclusion

This project demonstrates a complete end-to-end machine learning pipeline, from data preprocessing 

to model deployment readiness, following industry-level best practices in credit risk modeling.

---

## Plot Preview(Visualization)

Confusion Matrix
<br>
<br>

<img src="https://github.com/pratimadhende/Credit-risk-prediction/blob/ec247de80de65a100b3494bf2e8f174d937e9208/confusion_matrix.png" alt="Image Description" width="600">
<br>
<br>
Feature Importance
<br>
<br>
<img src="https://github.com/pratimadhende/Credit-risk-prediction/blob/ec247de80de65a100b3494bf2e8f174d937e9208/feature_importance.png" alt="Image Description" width="600">

---

## 💾 Model Saving
The trained model and scaler are saved using *Joblib*:
```text
credit_risk_model.pkl
scaler.pkl



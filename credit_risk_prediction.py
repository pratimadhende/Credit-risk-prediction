# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_auc_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

#Load Dataset
df=pd.read_csv("credit_risk_dataset.csv")
print(df.head())

# data exploration
print(df.info())
print(df.describe())
print(df.isnull().sum())

# handle missing values
num_col=df.select_dtypes(include=np.number).columns
df[num_col]=df[num_col].fillna(df[num_col].median())

cat_col=df.select_dtypes(include="object").columns
for col in cat_col:
    df[col]=df[col].fillna(df[col].mode()[0])

print("\nAfter handling missing values\n",df.isnull().sum())
# encode categorical variable
df=pd.get_dummies(df,drop_first=True)

# split features & target
X=df.drop("loan_status",axis=1)
y=df["loan_status"]

# train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# feature scaling
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# train multiple ML model 
# Logistic Regression 
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)

lr_pred=lr.predict(X_test)

# Random Forest
rf=RandomForestClassifier(n_estimators=200,random_state=42)
rf.fit(X_train,y_train)

rf_pred=rf.predict(X_test)

# support vector machine
svm=SVC(probability=True)
svm.fit(X_train,y_train)

svm_pred=svm.predict(X_test)

# model evalution function 
def evaluate_model(y_test,y_pred,y_prob,model_name):
    print(f"\n * {model_name}")
    print("Accuracy: ",accuracy_score(y_test,y_pred))
    print("ROC-AUC: ",roc_auc_score(y_test,y_prob))
    print(classification_report(y_test,y_pred))

# Evaluate all models 
evaluate_model(y_test,lr_pred,lr.predict_proba(X_test)[:,1],"Loagistric Regression")
evaluate_model(y_test,rf_pred,rf.predict_proba(X_test)[:,1],"Random Forest")
evaluate_model(y_test,svm_pred,svm.predict_proba(X_test)[:,1],"SVM")

# confusion matrix(random forest)
cm=confusion_matrix(y_test,rf_pred)
plt.figure()
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
plt.title("confusion matrix-random forest")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.savefig("confusion_matrix.png",dpi=300,bbox_inches="tight")
plt.show()

# ROC Curve 
rf_probs=rf.predict_proba(X_test)[:,1]
fpr,tpr,_=roc_curve(y_test,rf_probs)
plt.figure()
plt.plot(fpr,tpr,label="Random Forest")
plt.plot([0,1],[0,1],linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("roc_curve.png",dpi=300,bbox_inches="tight")
plt.show()

# feature importance
feature_importance=pd.Series(rf.feature_importances_,index=df.drop("loan_status",axis=1).columns).sort_values(ascending=False)
feature_importance.head(10).plot(kind="bar")
plt.title("Top 10 important feature")
plt.savefig("feature_importance.png",dpi=300,bbox_inches="tight")

# save final model 
joblib.dump(rf,"credit_risk_model.pkl")
joblib.dump(scaler,"scaler.pkl")
model=joblib.load("credit_risk_model.pkl")
print(type(model))
print(model.get_params())

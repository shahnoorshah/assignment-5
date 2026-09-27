import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

data = {
    "Study_Hours":[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
    "Attendance":[50, 55, 60, 58, 65, 62, 70, 68, 75, 72, 78, 80, 82, 85, 88, 86, 90, 92, 95, 96],
    "Pass":[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df=pd.DataFrame(data)
X=df[["Study_Hours", "Attendance"]]
y=df["Pass"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_probability=model.predict_proba(X_test)[:, 1]
fpr,tpr,thresholds=roc_curve(y_test,y_probability)
auc_score=roc_auc_score(y_test,y_probability)
print("AUC Score: ",auc_score)
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label="Logistic Regression (AUC={:.2f})".format(auc_score))
plt.plot([0,1],[0,1],linestyle="--",label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")

plt.legend()
plt.grid()
plt.show()
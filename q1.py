import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data={
    "Study_Hours":[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
    "Attendance":[50, 55, 60, 58, 65, 62, 70, 68, 75, 72, 78, 80, 82, 85, 88, 86, 90, 92, 95, 96],
    "Pass":[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df=pd.DataFrame(data)
print("Student Dataset:")
print(df)
X=df[["Study_Hours","Attendance"]]
y=df["Pass"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("\nModel Accuracy:",accuracy)
print("\nClassification Report:")
print(classification_report(y_test,y_pred))
study_hours=float(input("\nEnter study hours: "))
attendance=float(input("Enter attendance percentage: "))
new_student=pd.DataFrame({
    "Study_Hours":[study_hours],
    "Attendance":[attendance]
})
prediction=model.predict(new_student)
if prediction[0]==1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
probability=model.predict_proba(new_student)

print("Probability of Fail: ",round(probability[0][0]*100,2),"%")
print("Probability of Pass: ",round(probability[0][1]*100,2),"%")
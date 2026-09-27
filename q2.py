import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = {
    "Study_Hours":[1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
    "Attendance":[50, 55, 60, 58, 65, 62, 70, 68, 75, 72, 78, 80, 82, 85, 88, 86, 90, 92, 95, 96],
    "Pass": [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Student Dataset:")
print(df)
X=df[["Study_Hours", "Attendance"]]
y=df["Pass"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
probabilities=model.predict_proba(X_test)[:, 1]
thresholds=[0.3, 0.5, 0.7]
print("\nThreshold Comparison:")
print("-----------------------------------------")
print("Threshold\tPrecision\tRecall")
print("-----------------------------------------")
for threshold in thresholds:
    predictions=(probabilities >= threshold).astype(int)
    precision=precision_score(y_test, predictions, zero_division=0)
    recall=recall_score(y_test, predictions, zero_division=0)
    print(f"{threshold}\t\t{precision:.2f}\t\t{recall:.2f}")
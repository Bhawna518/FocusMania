import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/cleaned_productivity_data.csv")

encoder = LabelEncoder()
for col in ["Category", "Priority", "Mood"]:
    df[col] = encoder.fit_transform(df[col])

X = df[["Time_Spent_Min", "Category", "Priority", "Mood"]]
y = df["Completed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("🔹 Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("🔹 Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nRandom Forest Report:\n")
print(classification_report(y_test, rf_pred))
import matplotlib.pyplot as plt

feature_names = X.columns
importances = rf.feature_importances_

plt.figure(figsize=(6,4))
plt.barh(feature_names, importances)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.show()
00
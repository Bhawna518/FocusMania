import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/cleaned_productivity_data.csv")
sns.set(style="whitegrid")

plt.figure(figsize=(8,5))
sns.barplot(x="Category", y="Completed", data=df)
plt.title("Completion Rate by Category")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Time_Spent_Min"], kde=True)
plt.title("Time Spent Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x="Priority", y="Completed", data=df)
plt.title("Completion by Priority")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x="Mood", hue="Completed", data=df)
plt.title("Mood Impact on Task Completion")
plt.show()

print("✅ EDA completed successfully")

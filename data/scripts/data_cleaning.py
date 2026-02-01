import pandas as pd

RAW_PATH = "data/productivity_data.csv"
CLEAN_PATH = "data/cleaned_productivity_data.csv"

def clean_data():
    df = pd.read_csv(RAW_PATH)

    df = df.drop_duplicates()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for col in ["Task", "Category", "Priority", "Mood"]:
        df[col] = df[col].astype(str).str.strip().str.title()

    df["Completed"] = df["Completed"].map({"Yes": 1, "No": 0})

    df = df.ffill()

    df.to_csv(CLEAN_PATH, index=False)
    print("✅ Data cleaned and saved successfully")

if __name__ == "__main__":
    clean_data()

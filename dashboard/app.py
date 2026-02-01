import streamlit as st
import altair as alt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FocusMania | Productivity Intelligence",
    page_icon="📊",
    layout="wide"
)

sns.set_style("whitegrid")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/cleaned_productivity_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ FocusMania")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Analytics", "ML Insights", "Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered Productivity Platform")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h2 style='margin-bottom:0;'>📊 FocusMania</h2>
    <p style='color:grey;margin-top:0;'>Clean, Explainable & Actionable Productivity Analytics</p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ========================= OVERVIEW =========================
if page == "Overview":

    st.subheader("📌 Key Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tasks", len(df))
    c2.metric("Completion Rate", f"{df['Completed'].mean()*100:.1f}%")
    c3.metric("Avg Time Spent", f"{df['Time_Spent_Min'].mean():.0f} min")
    c4.metric("High Priority Tasks", (df["Priority"] == "High").sum())

    st.info(
        "FocusMania helps analyze productivity behavior, identify risks, "
        "and predict task completion using data analytics and machine learning."
    )

# ========================= ANALYTICS =========================
elif page == "Analytics":

    st.subheader("📈 Productivity Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.barplot(x="Category", y="Completed", data=df, ax=ax)
        ax.set_title("Completion by Category", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(x="Mood", hue="Completed", data=df, ax=ax)
        ax.set_title("Mood vs Completion", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.markdown("### ⏳ Productivity Trend Over Time")

    daily = df.groupby("Date")["Completed"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6,3))
    sns.lineplot(x="Date", y="Completed", data=daily, marker="o", ax=ax)
    ax.set_ylabel("Completion Rate")
    ax.set_xlabel("")
    st.pyplot(fig)

# ========================= ML INSIGHTS =========================
elif page == "ML Insights":

    st.subheader("🧠 Machine Learning Insights")

    df_ml = df.copy()
    encoder = LabelEncoder()
    for col in ["Category", "Priority", "Mood"]:
        df_ml[col] = encoder.fit_transform(df_ml[col])

    X = df_ml[["Time_Spent_Min", "Category", "Priority", "Mood"]]
    y = df_ml["Completed"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.metric("Final Model Used", "Random Forest")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
    ax.set_title("Feature Importance", fontsize=10)
    st.pyplot(fig)

# ========================= PREDICTION =========================
elif page == "Prediction":

    st.subheader("🔮 Predict Task Completion")

    df_ml = df.copy()
    encoder = LabelEncoder()
    for col in ["Category", "Priority", "Mood"]:
        df_ml[col] = encoder.fit_transform(df_ml[col])

    X = df_ml[["Time_Spent_Min", "Category", "Priority", "Mood"]]
    y = df_ml["Completed"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    col1, col2 = st.columns(2)

    with col1:
        time = st.slider("⏱ Time Spent (minutes)", 10, 180, 60)
        priority = st.selectbox("🔥 Priority", ["Low", "Medium", "High"])

    with col2:
        mood = st.selectbox("🧠 Mood", ["Focused", "Calm", "Tired", "Stressed"])
        category = st.selectbox("📂 Category", df["Category"].unique())

    priority_map = {"Low": 0, "Medium": 1, "High": 2}
    mood_map = {"Focused": 0, "Calm": 1, "Tired": 2, "Stressed": 3}
    category_map = {cat: i for i, cat in enumerate(df["Category"].unique())}

    if st.button("Predict Completion"):
        input_df = pd.DataFrame([[
            time,
            category_map[category],
            priority_map[priority],
            mood_map[mood]
        ]], columns=X.columns)

        prediction = model.predict(input_df)[0]

        # ---- Risk Score ----
        risk = 0
        if priority == "High":
            risk += 0.4
        if mood in ["Tired", "Stressed"]:
            risk += 0.4
        if time > 120:
            risk += 0.2

        risk = round(risk, 2)

        if prediction == 1:
            st.success("✅ High chance of task completion")
        else:
            st.warning("⚠️ Task may not be completed")

        st.metric("Risk Score", risk)

        if risk >= 0.7:
            st.info("💡 Recommendation: Break the task into smaller parts and take short breaks.")
        elif risk >= 0.4:
            st.info("💡 Recommendation: Try changing environment or rescheduling the task.")
        else:
            st.info("💡 Recommendation: Task looks safe. Maintain current approach.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("FocusMania • Industry-Level Analytics • Clean UI • Explainable ML")

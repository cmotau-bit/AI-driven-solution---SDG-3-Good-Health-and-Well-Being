# app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, names=columns, na_values="?")
    df = df.dropna()
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

# Preprocessing
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# UI
st.title("Heart Disease Risk Prediction")
st.subheader("Enter your health information:")

age = st.slider("Age", 29, 77, 54)
sex = st.radio("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
chol = st.slider("Cholesterol", 100, 600, 245)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.radio("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (3 = normal; 6 = fixed defect; 7 = reversible)", [3, 6, 7])

# Predict
user_data = np.array([
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]).reshape(1, -1)

user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)

# Output
if st.button("Predict"):
    if prediction[0] == 1:
        st.error("You may be at risk of heart disease.")
    else:
        st.success("You are likely not at risk of heart disease.")

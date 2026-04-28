import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Title
st.title("🎯 Student Placement Prediction System")

# Data Setup
data = {
    "CGPA": [8.2, 7.5, 6.8, 9.1, 5.9, 8.7, 7.0, 6.2, 8.9, 7.8],
    "IQ": [120, 110, 95, 130, 85, 125, 100, 90, 128, 115],
    "Communication": [1, 0, 0, 1, 2, 1, 0, 2, 1, 1], # Good:1, Avg:0, Poor:2
    "Internship": [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],    # Yes:1, No:0
    "Projects": [3, 2, 1, 4, 1, 3, 2, 1, 4, 3],
    "Aptitude": [80, 72, 60, 90, 50, 85, 68, 55, 88, 78],
    "Placed": [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
}
df = pd.DataFrame(data)

# Model Training
X = df.drop("Placed", axis=1)
y = df["Placed"]
model = LogisticRegression()
model.fit(X, y)

# User Input
st.subheader("Enter Student Details:")
cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
iq = st.number_input("IQ Score", 50, 200, 100)
comm = st.selectbox("Communication", ["Average", "Good", "Poor"])
intern = st.selectbox("Internship", ["No", "Yes"])
proj = st.slider("Projects", 0, 10, 2)
apt = st.slider("Aptitude Score", 0, 100, 70)

# Mappings
comm_dict = {"Average": 0, "Good": 1, "Poor": 2}
int_dict = {"No": 0, "Yes": 1}

if st.button("Predict Results"):
    features = np.array([[cgpa, iq, comm_dict[comm], int_dict[intern], proj, apt]])
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("✅ Prediction: Likely to be Placed!")
    else:
        st.error("❌ Prediction: Not Likely to be Placed.")

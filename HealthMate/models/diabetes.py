import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset from a local CSV file
@st.cache_data
def load_data():
    # Make sure to upload your `diabetes.csv` file into the Streamlit app.
    data = pd.read_csv("diabetes.csv")
    return data

# Train the logistic regression model
@st.cache_resource
def train_model(data):
    X = data.iloc[:, :-1]  # All columns except the last one (Outcome)
    y = data.iloc[:, -1]   # The last column is the target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit app
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict the likelihood of diabetes:")

# Load data and train model
data = load_data()
model, X_test, y_test = train_model(data)

# Ensure y_test and predictions are consistent
y_test = y_test.astype(int)
y_pred = model.predict(X_test).astype(int)

# Accuracy Display
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0, step=0.1)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, step=0.1)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, step=0.1)
insulin = st.number_input("Insulin Level", min_value=0.0, step=0.1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Prepare the input for prediction
    user_input = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, diabetes_pedigree_function, age
    ]])

    # Make a prediction
    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    # Display the prediction
    if prediction[0] == 1:
        st.success(f"The model predicts that the individual is likely to have diabetes. "
                   f"Probability: {prediction_proba[0][1]:.2f}")
    else:
        st.success(f"The model predicts that the individual is unlikely to have diabetes. "
                   f"Probability: {prediction_proba[0][0]:.2f}")

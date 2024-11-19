import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load and preprocess dataset
df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
df['sex'] = df.sex.map({0: 'female', 1: 'male'}).map({'female': 0, 'male': 1})

# Define features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-Test split and scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    "SVM": SVC(kernel='rbf', probability=True).fit(X_train, y_train),
    "Random Forest": RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train),
    "Naive Bayes": GaussianNB().fit(X_train, y_train),
    "Logistic Regression": LogisticRegression().fit(X_train, y_train),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
}

# Streamlit App
st.title("Heart Disease Prediction")

# User inputs
st.sidebar.header("Input Features")
def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', options=['Male', 'Female'], index=0)
    cp = st.sidebar.slider('Chest Pain Type (1-4)', 1, 4, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure (mmHg)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], index=0)
    restecg = st.sidebar.slider('Resting ECG (0-2)', 0, 2, 0)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 200, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', options=[0, 1], index=0)
    oldpeak = st.sidebar.slider('ST Depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.slider('Slope (1-3)', 1, 3, 2)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (3, 6, 7)', 3, 7, 3)
    data = {
        'age': age, 'sex': 1 if sex == 'Male' else 0, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Scale user input
input_scaled = scaler.transform(input_df)

# Predictions
st.subheader("Predictions")
selected_model = st.selectbox("Select Model", options=list(models.keys()))

if st.button("Predict"):
    model = models[selected_model]
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0][1] if selected_model != "SVM" else None
    st.write(f"Prediction: {'Heart Disease Detected' if prediction[0] == 1 else 'No Heart Disease'}")
    if prediction_proba is not None:
        st.write(f"Prediction Probability: {prediction_proba:.2f}")

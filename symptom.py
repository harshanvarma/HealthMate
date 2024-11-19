import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache_data
def load_data():
    # Replace with your actual file paths
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")

    # Symptom list
    l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
          'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
          # ... (rest of the symptoms from the original list)
          'yellow_crust_ooze']

    disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
               'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 
               # ... (rest of the diseases from the original list)
               'Impetigo']

    # Use Label Encoder to convert target to integers
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])
    tr['prognosis'] = le.transform(tr['prognosis'])

    X = df[l1]
    y = df['prognosis']
    X_test = tr[l1]
    y_test = tr['prognosis']

    return l1, le.classes_, X, y, X_test, y_test

# Train models
@st.cache_data
def train_models(X, y, X_test, y_test):
    # Decision Tree
    clf3 = tree.DecisionTreeClassifier()
    clf3.fit(X, y)
    dt_accuracy = accuracy_score(y_test, clf3.predict(X_test))

    # Random Forest
    clf4 = RandomForestClassifier()
    clf4.fit(X, y)
    rf_accuracy = accuracy_score(y_test, clf4.predict(X_test))

    # Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X, y)
    nb_accuracy = accuracy_score(y_test, gnb.predict(X_test))

    return clf3, clf4, gnb, dt_accuracy, rf_accuracy, nb_accuracy

# Predict disease
def predict_disease(symptoms, l1, disease, model):
    l2 = [0] * len(l1)
    for k, symptom in enumerate(l1):
        if symptom in symptoms:
            l2[k] = 1

    predict = model.predict([l2])
    predicted = predict[0]
    
    return disease[predicted]

def main():
    st.title("Disease Predictor using Machine Learning")

    # Load data and train models
    l1, disease, X, y, X_test, y_test = load_data()
    dt_model, rf_model, nb_model, dt_acc, rf_acc, nb_acc = train_models(X, y, X_test, y_test)

    # Patient name input
    patient_name = st.text_input("Name of the Patient")

    # Symptom selection
    st.subheader("Select Symptoms")
    selected_symptoms = st.multiselect(
        "Choose up to 5 symptoms", 
        sorted(l1), 
        max_selections=5
    )

    # Prediction button
    if st.button("Predict"):
        if selected_symptoms:
            # Decision Tree Prediction
            dt_prediction = predict_disease(selected_symptoms, l1, disease, dt_model)
            st.success(f"Decision Tree Prediction: {dt_prediction}")
            st.info(f"Decision Tree Model Accuracy: {dt_acc:.2%}")

            # Random Forest Prediction
            rf_prediction = predict_disease(selected_symptoms, l1, disease, rf_model)
            st.success(f"Random Forest Prediction: {rf_prediction}")
            st.info(f"Random Forest Model Accuracy: {rf_acc:.2%}")

            # Naive Bayes Prediction
            nb_prediction = predict_disease(selected_symptoms, l1, disease, nb_model)
            st.success(f"Naive Bayes Prediction: {nb_prediction}")
            st.info(f"Naive Bayes Model Accuracy: {nb_acc:.2%}")
        else:
            st.warning("Please select symptoms first!")

if __name__ == "__main__":
    main()
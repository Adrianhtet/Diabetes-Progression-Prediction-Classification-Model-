import streamlit as st
import pickle
import pandas as pd

st.set_page_config(
    page_title="Diabetes Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Prediction (Classification Model)")

st.write(
    "Enter the patient's information below to predict whether "
    "the patient is likely to have diabetes."
)


with st.sidebar:

    st.header("Patient Information")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"]
    )

    age = st.slider(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=1.0
    )

    st.divider()

    st.subheader("Medical History")

    hypertension_choice = st.selectbox(
        "Hypertension",
        ["No", "Yes"]
    )

    hypertension = 1 if hypertension_choice == "Yes" else 0

    heart_disease_choice = st.selectbox(
        "Heart Disease",
        ["No", "Yes"]
    )

    heart_disease = 1 if heart_disease_choice == "Yes" else 0

    smoking_history = st.selectbox(
        "Smoking History",
        [
            "No Info",
            "never",
            "former",
            "current",
            "not current",
            "ever"
        ]
    )

    st.divider()

    st.subheader("Body & Blood Measurements")

    bmi = st.slider(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    hba1c = st.slider(
        "HbA1c Level",
        min_value=3.0,
        max_value=10.0,
        value=5.5,
        step=0.1
    )

    blood_glucose = st.slider(
        "Blood Glucose Level",
        min_value=50,
        max_value=300,
        value=120,
        step=1
    )


predict_button = st.button("Predict")


if predict_button:

    data = pd.DataFrame(
        [[
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            hba1c,
            blood_glucose
        ]],
        columns=[
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "smoking_history",
            "bmi",
            "HbA1c_level",
            "blood_glucose_level"
        ]
    )

    with open("diabetes_classification_model.pkl", "rb") as file:
        model = pickle.load(file)

    result = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if result == 1:

        st.error("Diabetes Detected")

        st.metric(
            "Estimated Diabetes Probability",
            f"{probability * 100:.2f}%"
        )

    else:

        st.success("No Diabetes Detected")

        st.metric(
            "Estimated Diabetes Probability",
            f"{probability * 100:.2f}%"
        )
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model_path = "/content/titanic_model.pkl"
scaler_path = "/content/scaler.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=0.0, format="%.1f")
fare = st.number_input("Fare", min_value=0.0, format="%.2f")
embarked = st.selectbox("Port of Embarkation", ["C - Cherbourg", "Q - Queenstown", "S - Southampton"])

# Convert categorical inputs to numerical values
sex_numeric = 1 if sex == "Female" else 0
embarked_mapping = {"C - Cherbourg": 0, "Q - Queenstown": 1, "S - Southampton": 2}
embarked_numeric = embarked_mapping[embarked]

# Predict button
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_numeric, age, fare, embarked_numeric]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    result = "Survived 🎉" if prediction[0] == 1 else "Did not survive 😢"
    st.success(f'Prediction: **{result}**')
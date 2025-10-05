import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf

# Load the trained model'
model = tf.keras.models.load_model('ann_model.h5')
# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the encoder
with open('ohe_geo.pkl', 'rb') as f:
    encoder = pickle.load(f)
# Load the label encoder for the target variable
with open('le_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict if they will leave the bank.")
# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
age = st.slider("Age", min_value=18, max_value=100, value=30)
tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=2)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("select gender", ['Male', 'Female'])
# Encode categorical variables
input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender":gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}
geo_encoded = pd.DataFrame(encoder.transform([[geography]]).toarray(), columns=encoder.get_feature_names_out(['Geography']))

# Convert input_data dictionary to DataFrame and concatenate with geo_encoded
input_data = pd.DataFrame([input_data])
input_data = pd.concat([input_data, geo_encoded], axis=1)
input_data = input_data.drop('Geography', axis=1)
input_data.Gender = label_encoder.transform(input_data.Gender)

scaled_input_data = scaler.transform(input_data)
prediction = model.predict(scaled_input_data)
if prediction[0][0] > 0.5:
    st.write("The customer is likely to leave the bank.")
else:   
    st.write("The customer is likely to stay with the bank.")


import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Enter the house area to predict the house price.")

# User input
area = st.number_input(
    "House Area (Square Feet)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(np.array([[area]]))

    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")
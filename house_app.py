import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(page_title="House Price Prediction", page_icon=":house:", layout="centered")

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please ensure it exists in the directory.")

local_css("style.css")

# Load the entire pipeline (including preprocessing and the model)
model = pickle.load(open('best_model.pkl', 'rb'))

def predict_price(area, location, size, sqft, bath, balcony):
    # Create a DataFrame with the correct column names
    input_data = pd.DataFrame(
        data=[[area, location, size, sqft, bath, balcony]],
        columns=['area_type', 'location', 'size', 'total_sqft', 'bath', 'balcony']
    )

    # Make prediction using the pipeline
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title("House Price Prediction")

st.markdown("""
Welcome to the House Price Prediction app. Please provide the details of the house to get an estimated price.
""")

# Input fields
st.header("Enter House Details")

location = st.text_input("Location", help="Enter the location of the house")
area = st.text_input("Area", help="Enter the area (e.g., neighborhood) of the house")
sqft = st.number_input("Square Feet", min_value=100, help="Enter the total square feet of the house (minimum 100)")
bath = st.number_input("Number of Bathrooms", min_value=1, help="Enter the number of bathrooms (minimum 1)")
balcony = st.number_input("Number of Balconies", min_value=1, help="Enter the number of balconies (minimum 1)")
size = st.number_input("Size (in BHK)", min_value=1, help="Enter the size in terms of BHK (e.g., 2 for 2BHK, minimum 1)")

# Predict button
if st.button("Predict"):
    if location and area:
        prediction = predict_price(area, location, size, sqft, bath, balcony)
        st.success(f"Your house estimate price is ₹ {prediction:.2f} lakhs")
    else:
        st.error("Please provide all the required inputs.")

st.markdown("""
---
**Note**: The prediction provided is an estimate and may not reflect the actual market price.
""")



import numpy as np
import pickle
import streamlit as st

# Set page configuration
st.set_page_config(page_title="House Price Prediction", page_icon=":house:", layout="centered")

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
            print(css)  # Debug: Print CSS content to console
    except FileNotFoundError:
        st.error(f"CSS file '{file_name}' not found. Please ensure it exists in the directory.")

local_css("style.css")

# Load the model and dictionaries
model = pickle.load(open('model.pkl', 'rb'))
# index_dict = pickle.load(open('cat', 'rb'))
# location_cat = pickle.load(open('location_cat', 'rb'))

def predict_price(location, area, sqft, bath, balcony, size):
    new_vector = np.zeros(152)

    # # Process location
    # if location not in location_cat:
    #     new_vector[146] = 1
    # else:
    #     new_vector[index_dict[str(location)]] = 1

    # # Process area
    # new_vector[index_dict[str(area)]] = 1

    # Other features
    new_vector[0] = sqft
    new_vector[1] = bath
    new_vector[2] = balcony
    new_vector[3] = size

    # Debug: Print new_vector to verify its values
    print("Feature vector for prediction:", new_vector)

    # Make prediction
    new = [new_vector]
    prediction = model.predict(new)
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
        prediction = predict_price(location, area, sqft, bath, balcony, size)
        st.success(f"Your house estimate price is ₹ {prediction:.2f} lakhs")
    else:
        st.error("Please provide all the required inputs.")

st.markdown("""
---
**Note**: The prediction provided is an estimate and may not reflect the actual market price.
""")

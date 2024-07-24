import streamlit as st
import joblib
import pandas as pd

# Ganti dengan jalur absolut ke file model Anda
import os
import joblib

# Tentukan jalur absolut untuk file model
model_path = os.path.join('/mount/src/mpml', 'best_model.pkl')
print(f"Attempting to load model from: {model_path}")

if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")

model = joblib.load(model_path)

import os
import joblib

model_path = os.path.join('/mount/src/mpml', 'best_model.pkl')
print(f"Attempting to load model from: {model_path}")

if os.path.isfile(model_path):
    print(f"File {model_path} found.")
else:
    print(f"File {model_path} does not exist.")
 
# Streamlit application
def main():
    st.title('Welcome to the Customer App')

    # Form for input
    with st.form(key='prediction_form'):
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Prefer Not to Say'])
        occupation = st.selectbox('Occupation', ['Employee', 'Student', 'Self Employed', 'House Wife', 'Other'])
        monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
        educational_qualifications = st.selectbox('Educational Qualifications', ['Graduate', 'Post Graduate', 'Ph.D', 'School', 'Uneducated'])
        feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
        age = st.number_input('Age', min_value=0)
        family_size = st.number_input('Family Size', min_value=1, max_value=10)
        latitude = st.number_input('Latitude')
        longitude = st.number_input('Longitude')
        pin_code = st.number_input('Pin Code')

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Create DataFrame for prediction
            data = pd.DataFrame({
                'Gender': [gender],
                'Marital Status': [marital_status],
                'Occupation': [occupation],
                'Monthly Income': [monthly_income],
                'Educational Qualifications': [educational_qualifications],
                'Feedback': [feedback],
                'Age': [age],
                'Family size': [family_size],
                'latitude': [latitude],
                'longitude': [longitude],
                'Pin code': [pin_code]
            })

            # Predict
            prediction = model.predict(data)[0]
            st.write(f'Prediction: {prediction}')

if __name__ == "__main__":
    main()

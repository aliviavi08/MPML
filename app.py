import streamlit as st
import joblib
import pandas as pd
import os

# Load the model
model_path = os.path.join('/mount/src/mpml', 'best_model.pkl')
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
model = joblib.load(model_path)

# Streamlit application
def main():
    st.title('Customer Prediction App')

    # Form for input
    with st.form(key='prediction_form'):
        age = st.number_input('Age', min_value=0)  # Pastikan fitur Age ada karena disebutkan dalam error
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital_Status', ['Single', 'Married', 'Prefer Not to Say'])
        occupation = st.selectbox('Occupation', ['Employee', 'Student', 'Self Employed', 'House Wife', 'Other'])
        monthly_income = st.selectbox('Monthly_Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
        educational_qualifications = st.selectbox('Educational_Qualifications', ['Graduate', 'Post Graduate', 'Ph.D', 'School', 'Uneducated'])
        feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
        family_size = st.number_input('Family_Size', min_value=1, max_value=10)
        latitude = st.number_input('Latitude')
        longitude = st.number_input('Longitude')
        pin_code = st.number_input('Pin_Code')

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Convert inputs into a DataFrame with correct column names
            data = pd.DataFrame({
                'Age': [age],  # Pastikan fitur Age ada
                'Gender': [gender],
                'Marital_Status': [marital_status],
                'Occupation': [occupation],
                'Monthly_Income': [monthly_income],
                'Educational_Qualifications': [educational_qualifications],
                'Feedback': [feedback],
                'Family_Size': [family_size],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Pin_Code': [pin_code]
            })

            # Predict
            try:
                prediction = model.predict(data)[0]
                st.write(f'Prediction: {prediction}')
            except ValueError as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()

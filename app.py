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
        gender = st.selectbox('Gender', ['Male', 'Female'])
        marital_status = st.selectbox('Marital_Status', ['Single', 'Married', 'Prefer Not to Say'])
        occupation = st.selectbox('Occupation', ['Employee', 'Student', 'Self Employed', 'House Wife', 'Other'])
        monthly_income = st.selectbox('Monthly_Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
        educational_qualifications = st.selectbox('Educational Qualifications', ['Graduate', 'Post Graduate', 'Ph.D', 'School', 'Uneducated'])
        feedback = st.selectbox('Feedback', ['Positive', 'Negative'])
        age = st.number_input('Age', min_value=0)
        family_size = st.number_input('Family_Size', min_value=1, max_value=10)
        latitude = st.number_input('Latitude')
        longitude = st.number_input('Longitude')
        pin_code = st.number_input('Pin_Code')

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Mapping for categorical variables
            gender_map = {'Male': 0, 'Female': 1}
            marital_status_map = {'Single': 0, 'Married': 1, 'Prefer Not to Say': 2}
            occupation_map = {'Employee': 0, 'Student': 1, 'Self Employed': 2, 'House Wife': 3, 'Other': 4}
            income_map = {'No Income': 0, 'Below Rs.10000': 1, '10001 to 25000': 2, '25001 to 50000': 3, 'More than 50000': 4}
            education_map = {'Graduate': 0, 'Post Graduate': 1, 'Ph.D': 2, 'School': 3, 'Uneducated': 4}
            feedback_map = {'Positive': 1, 'Negative': 0}

            # Convert to numerical values
            data = pd.DataFrame({
                'Gender': [gender_map[gender]],
                'Marital_Status': [marital_status_map[marital_status]],
                'Occupation': [occupation_map[occupation]],
                'Monthly_Income': [income_map[monthly_income]],
                'Educational_Qualifications': [education_map[educational_qualifications]],
                'Feedback': [feedback_map[feedback]],
                'Age': [age],
                'Family_Size': [family_size],
                'Latitude': [latitude],
                'Longitude': [longitude],
                'Pin_Code': [pin_code]
            })

            # Ensure the columns are in the correct order as per training data
            expected_features = ['Gender', 'Marital_Status', 'Occupation', 'Monthly_Income',
                                 'Educational_Qualifications', 'Feedback', 'Age', 'Family_Size',
                                 'Latitude', 'Longitude', 'Pin_Code']
            data = data[expected_features]

            # Predict
            try:
                prediction = model.predict(data)[0]
                st.write(f'Prediction: {prediction}')
            except ValueError as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()

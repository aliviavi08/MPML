import streamlit as st
import joblib
import pandas as pd
import os

# Load the model
model_path = 'best_model.pkl'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
model = joblib.load(model_path)

# Streamlit application
def main():
    st.title('Customer Prediction App')

    # Form for input
    with st.form(key='prediction_form'):
        age = st.number_input('Age', min_value=0)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
        family_size = st.number_input('Family Size', min_value=1, max_value=10)

        submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Convert inputs into a DataFrame with correct column names
            gender_map = {'Male': 0, 'Female': 1}
            income_map = {
                'No Income': 0,
                'Below Rs.10000': 1,
                '10001 to 25000': 2,
                '25001 to 50000': 3,
                'More than 50000': 4
            }

            data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender_map[gender]],
                'Monthly_Income': [income_map[monthly_income]],
                'Family_Size': [family_size]
            })

            # Predict
            prediction = model.predict(data)[0]
            st.write(f'Prediction: {"Yes" if prediction == 1 else "No"}')

if __name__ == "__main__":
    main()

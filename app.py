import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_filename = '/mnt/data/cacs_tot_prediction_model.joblib'
model = joblib.load(model_filename)

# Function to make predictions
def predict_cacs(gender, age, previous_mi_cabg):
    # Encode inputs
    gender_encoded = 1 if gender == 'Male' else 0
    previous_mi_cabg_encoded = 1 if previous_mi_cabg == 'Yes' else 0
    
    # Create input array
    X = np.array([[gender_encoded, age, previous_mi_cabg_encoded]])
    
    # Make prediction
    prediction = model.predict(X)
    return prediction[0]

# Streamlit app
st.title('Dr Peturssons CACS Prediction App')
st.write('Enter the following details to predict CACS')

# Input fields
age = st.number_input('Age', min_value=0, max_value=120, value=50)
gender = st.selectbox('Gender', ['Male', 'Female'])
previous_mi_cabg = st.selectbox('Previous MI/CABG/PCI/Stent', ['Yes', 'No'])

# Predict button
if st.button('Predict'):
    cacs_prediction = predict_cacs(gender, age, previous_mi_cabg)
    
    # Display the result in a styled card
    st.markdown(
        f"""
        <div style="background-color: #f0f0f5; padding: 10px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <h2 style="color: #4CAF50;">Predicted CACS</h2>
            <p style="font-size: 24px; color: #000000;"><b>{cacs_prediction:.2f}</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the app
# Save this file as app.py and run `streamlit run app.py` in the terminal
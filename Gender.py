import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st
import os
import warnings

warnings.filterwarnings("ignore")

# Loading the saved models
model = pk.load(open('Gender.sav', 'rb'))

# Sidebar for navigation
st.sidebar.title("Navigation")
selected = st.sidebar.selectbox("Choose the prediction model", ["Gender Prediction"])

# Gender Prediction Page
if selected == 'Gender Prediction':
    st.markdown("<h1 style='text-decoration: underline;'>Gender Prediction using ML</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        forehead_width_cm = st.text_input('Forehead Width (cm)')
        nose_wide = st.text_input('Nose Wide')
        lips_thin = st.text_input('Lips Thin')
    with col2:
        forehead_height_cm = st.text_input('Forehead Height (cm)')
        nose_long = st.text_input('Nose Long')
        distance_nose_to_lip_long = st.text_input('Distance from Nose to Lip (long)')
    
    # Code for Prediction
    Predict_diagnosis = ''
    if st.button('Gender Test Result'):
        try:
            # Convert input to float
            input_data = [
                float(forehead_width_cm),
                float(forehead_height_cm),
                float(nose_wide),
                float(nose_long),
                float(lips_thin),
                float(distance_nose_to_lip_long)
            ]
            gen_prediction = model.predict([input_data])
        
            if gen_prediction[0] == 0:
                Predict_diagnosis = 'It is a Female'
            else:
                Predict_diagnosis = 'It is a Male'
        except ValueError:
            st.error("Please enter valid numerical values for all fields.")
    
    st.success(Predict_diagnosis)

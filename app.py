import streamlit as st
import numpy as np
import joblib

# Load saved model and scalers
model = joblib.load('model.pkl')
stand_scaler = joblib.load('standscaler.pkl')
minmax_scaler = joblib.load('minmaxscaler.pkl')

# Crop mapping
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Streamlit App UI
st.title('🌱 Crop Recommendation System')

N = st.number_input('Nitrogen (N)', min_value=0, max_value=140, value=50)
P = st.number_input('Phosphorus (P)', min_value=5, max_value=145, value=50)
K = st.number_input('Potassium (K)', min_value=5, max_value=205, value=50)
temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (%)', min_value=10.0, max_value=100.0, value=80.0)
ph = st.number_input('pH value', min_value=3.0, max_value=10.0, value=6.5)
rainfall = st.number_input('Rainfall (mm)', min_value=20.0, max_value=300.0, value=100.0)

if st.button('Predict Best Crop'):
    feature_list = [N, P, K, temperature, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = stand_scaler.transform(single_pred)
    scaled_features = minmax_scaler.transform(scaled_features)

    prediction = model.predict(scaled_features)[0]
    crop_name = crop_dict.get(prediction, "Unknown")

    st.success(f"The recommended crop is: 🌾 {crop_name}")

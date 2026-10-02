
import streamlit as st
import joblib
import pandas as pd

st.title("🌱 Tobacco Yield Prediction App")
st.write("Enter the field parameters below to predict the tobacco yield.")

@st.cache_resource
def load_model():
    return joblib.load("tobacco_yield_model.joblib")

model = load_model()

# --- CREATE INPUT FIELDS (Matching the exact names from the error) ---
planting_day = st.number_input("Planting Day of Year", min_value=1, max_value=366, value=150)
rainfall = st.number_input("Rainfall (mm)", value=800.0)
pesticide = st.number_input("Pesticide Applications", min_value=0, value=3)
humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)

# ⚠️ IMPORTANT: Change these options to match your actual dataset!
province = st.selectbox("Province", ["Harare", "Mashonaland", "Manicaland", "Midlands", "Masvingo"]) 

# ⚠️ IMPORTANT: If your model expects 0/1 instead of Yes/No, change this to [0, 1]
irrigation = st.selectbox("Irrigation", ["Yes", "No"])

mechanization = st.number_input("Mechanization Index", value=0.5)
avg_temp = st.number_input("Average Temperature (°C)", value=25.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", value=100.0)
extension_visits = st.number_input("Extension Visits", min_value=0, value=2)

if st.button("Predict Yield"):
    
    # ⚠️ CRITICAL: The keys in this dictionary MUST exactly match the column names 
    # your model was trained on.
    input_data = {
        'planting_day_of_year': [planting_day],
        'rainfall_mm': [rainfall],
        'pesticide_applications': [pesticide],
        'relative_humidity_pct': [humidity],
        'province': [province],
        'irrigation': [irrigation],
        'mechanization_index': [mechanization],
        'avg_temperature_c': [avg_temp],
        'fertilizer_kg_per_ha': [fertilizer],
        'extension_visits': [extension_visits]
    }
    
    features_df = pd.DataFrame(input_data)
    
    try:
        prediction = model.predict(features_df)[0]
        st.success(f"🌾 Predicted Tobacco Yield: **{prediction:.2f}** kg/ha")
    except Exception as e:
        st.error(f"Error: {e}")
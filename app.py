import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SaveFood - Decision Intelligence", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('savefood_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

try:
    model, model_columns = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

st.title("🥗 SaveFood: Predict, Prepare & Prevent")
st.markdown("Decision Intelligence System for College Canteen Food Waste Reduction")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Operational Parameters")
    attendance = st.number_input("Expected Attendance Count", min_value=100, max_value=3000, value=1150, step=10)
    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    weather = st.selectbox("Forecasted Weather", ["Clear", "Rainy", "Hot"])
    is_event = st.radio("Special Campus Event / Fest?", ["No", "Yes"])

is_event_val = 1 if is_event == "Yes" else 0
input_dict = {
    'attendance_count': attendance,
    'is_event': is_event_val,
    'day_of_week_' + day_of_week: 1,
    'weather_' + weather: 1
}

input_df = pd.DataFrame([input_dict])
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

with col2:
    st.subheader("💡 Decision Analytics")
    if st.button("Generate Demand Recommendation", type="primary"):
        predicted_demand = int(model.predict(input_df)[0])
        buffer_qty = int(predicted_demand * 1.05)
        est_waste_saved_kg = round((predicted_demand * 0.20) * 0.35, 1)

        st.metric(label="Predicted Portion Demand", value=f"{predicted_demand} Meals")
        st.metric(label="Recommended Preparation Target (5% Buffer)", value=f"{buffer_qty} Meals")
        
        st.success(f"**Action Plan:** Prepare exactly **{buffer_qty} portions**.")
        st.info(f"🌱 Estimated food waste prevented today: **~{est_waste_saved_kg} kg**")

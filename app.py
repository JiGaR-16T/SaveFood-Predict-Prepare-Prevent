import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="SaveFood - Decision Intelligence",
    page_icon="🥗",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    </style>
""", unsafe_allow_html=True)

# Load Artifacts and Data
@st.cache_resource
def load_artifacts():
    model = joblib.load('savefood_model.pkl')
    columns = joblib.load('model_columns.pkl')
    df = pd.read_csv('canteen_data.csv')
    return model, columns, df

try:
    model, model_columns, df = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found! Please run train_model.py first.")
    st.stop()

# Visual Banner Section
st.markdown("""
    <div style="background-color: #1E3A8A; padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 25px;">
        <h1 style="color: #FACC15; margin: 0; font-size: 2.5rem;">🥗 SaveFood: Demand Intelligence System</h1>
        <p style="font-size: 1.2rem; color: #E0E7FF; margin-top: 10px;">
            🍲 <b>Predict Consumption</b> &nbsp;|&nbsp; 📦 <b>Optimize Preparation</b> &nbsp;|&nbsp; 🌱 <b>Prevent Campus Wastage</b>
        </p>
    </div>
""", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📋 Input Operational Parameters")
    
    attendance = st.number_input("Expected Attendance Count", min_value=100, max_value=3000, value=1150, step=10)
    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    weather = st.selectbox("Forecasted Weather", ["Clear", "Rainy", "Hot"])
    is_event = st.radio("Special Campus Event / Fest?", ["No", "Yes"], horizontal=True)

    # Encode Input Data
    is_event_val = 1 if is_event == "Yes" else 0
    input_dict = {
        'attendance_count': attendance,
        'is_event': is_event_val,
        'day_of_week_' + day_of_week: 1,
        'weather_' + weather: 1
    }

    input_df = pd.DataFrame([input_dict])
    for c in model_columns:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[model_columns]

    predict_btn = st.button("🚀 Calculate Demand Recommendation", type="primary", use_container_width=True)

with col2:
    st.subheader("💡 Decision Analytics & Insights")
    
    if predict_btn:
        predicted_demand = int(model.predict(input_df)[0])
        buffer_qty = int(predicted_demand * 1.05)
        est_waste_saved_kg = round((predicted_demand * 0.20) * 0.35, 1)
        est_cost_saved = int(est_waste_saved_kg * 120)  # ~120 INR/kg estimated food cost

        m1, m2 = st.columns(2)
        m1.metric(label="Predicted Demand", value=f"{predicted_demand} Meals")
        m2.metric(label="Recommended Prep (5% Buffer)", value=f"{buffer_qty} Meals")

        m3, m4 = st.columns(2)
        m3.metric(label="Est. Waste Prevented", value=f"{est_waste_saved_kg} kg", delta="Saved Today")
        m4.metric(label="Est. Cost Saved", value=f"₹{est_cost_saved}", delta="Optimized")

        st.success(f"**Action Plan:** Prepare exactly **{buffer_qty} portions** to meet demand while preventing overproduction.")
    else:
        st.info("Adjust the operational parameters on the left and click **Calculate Demand Recommendation**.")

st.divider()

# Graphical Analytics Section
st.subheader("📊 Historical Performance Analytics")

chart_tab1, chart_tab2 = st.tabs(["📈 Attendance vs Portions Sold", "🌱 Food Waste History"])

with chart_tab1:
    st.markdown("**Historical Attendance vs. Food Demands (Past 30 Days)**")
    recent_df = df.tail(30).copy()
    st.line_chart(recent_df, x="date", y=["attendance_count", "portions_sold"], height=300)

with chart_tab2:
    st.markdown("**Historical Daily Food Waste Log (kg)**")
    st.bar_chart(recent_df, x="date", y="food_wasted_kg", height=300)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SaveFood - Canteen Intelligence", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #bae6fd 100%); background-attachment: fixed; }
    .hero-card { background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%); padding: 30px; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 25px rgba(2, 132, 199, 0.2); margin-bottom: 25px; border: 1px solid rgba(255, 255, 255, 0.3); }
    .stMetric { background: rgba(255, 255, 255, 0.75) !important; backdrop-filter: blur(10px); padding: 15px !important; border-radius: 15px !important; border: 1px solid #7dd3fc !important; box-shadow: 0 4px 12px rgba(14, 165, 233, 0.08); }
    .stButton>button { background: linear-gradient(90deg, #0284c7 0%, #0284c7 100%) !important; color: white !important; border-radius: 10px !important; font-weight: bold !important; border: none !important; transition: all 0.3s ease !important; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3); }
    .stButton>button:hover { background: linear-gradient(90deg, #0369a1 0%, #075985 100%) !important; transform: translateY(-2px); }
    </style>
""", unsafe_allow_html=True)

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

st.markdown("""
    <div class="hero-card">
        <h1 style="color: #e0f2fe; margin: 0; font-size: 2.6rem; font-weight: 700;">🥗 SaveFood: Demand Intelligence</h1>
        <p style="font-size: 1.15rem; color: #bae6fd; margin-top: 10px; letter-spacing: 0.5px;">
            🍲 <b>Predict Daily Consumption</b> &nbsp;|&nbsp; 📦 <b>Optimize Preparation</b> &nbsp;|&nbsp; 🌱 <b>Zero Waste Canteen</b>
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------- Prescriptive Recommendation Engine ----------
def recommend_action(predicted_demand, prep_quantity, cost_per_meal=50):
    surplus = prep_quantity - predicted_demand
    surplus_pct = surplus / prep_quantity if prep_quantity else 0
    if surplus_pct <= 0.05:
        return "✅ No Action Needed", "Prep closely matches predicted demand.", "success", surplus
    elif surplus_pct <= 0.15:
        return "🏷️ Offer Discount", "Small surplus — clear it with a 20% discount in the last hour.", "warning", surplus
    elif surplus_pct <= 0.30:
        return "🤝 Donate to NGO", "Moderate surplus — donate before spoilage instead of discarding.", "warning", surplus
    else:
        return "📉 Cut Next Batch by 20%", "Large surplus — adjust upstream prep planning.", "error", surplus

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📋 Input Operational Parameters")
    attendance = st.number_input("Expected Attendance Count", min_value=100, max_value=3000, value=1150, step=10)
    day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    weather = st.selectbox("Forecasted Weather", ["Clear", "Rainy", "Hot"])
    is_event = st.radio("Special Campus Event / Fest?", ["No", "Yes"], horizontal=True)

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

    if predict_btn:
        predicted_demand = int(model.predict(input_df)[0])
        st.session_state['predicted_demand'] = predicted_demand
        st.session_state['buffer_qty'] = int(predicted_demand * 1.05)
        st.session_state['est_waste_saved_kg'] = round((predicted_demand * 0.20) * 0.35, 1)
        st.session_state['est_cost_saved'] = int(st.session_state['est_waste_saved_kg'] * 120)

with col2:
    st.subheader("💡 Decision Analytics & Insights")

    if 'predicted_demand' in st.session_state:
        predicted_demand = st.session_state['predicted_demand']
        buffer_qty = st.session_state['buffer_qty']

        m1, m2 = st.columns(2)
        m1.metric(label="Predicted Demand", value=f"{predicted_demand} Meals")
        m2.metric(label="Recommended Prep (5% Buffer)", value=f"{buffer_qty} Meals")

        m3, m4 = st.columns(2)
        m3.metric(label="Est. Waste Prevented", value=f"{st.session_state['est_waste_saved_kg']} kg", delta="Saved Today")
        m4.metric(label="Est. Cost Saved", value=f"₹{st.session_state['est_cost_saved']}", delta="Optimized")

        st.success(f"**Action Plan:** Prepare exactly **{buffer_qty} portions** to meet demand while preventing overproduction.")
    else:
        st.info("Adjust the operational parameters on the left and click **Calculate Demand Recommendation**.")

st.divider()

# ---------- Prescriptive Recommendation Tab ----------
st.subheader("🎯 Prescriptive Recommendation Engine")

if 'predicted_demand' in st.session_state:
    predicted_demand = st.session_state['predicted_demand']
    prep_quantity = st.number_input(
        "Actual Planned Prep Quantity (what staff intends to cook)",
        min_value=0, value=st.session_state['buffer_qty'], step=10
    )
    action, reason, level, surplus = recommend_action(predicted_demand, prep_quantity)
    savings = max(surplus, 0) * 50  # ₹50/meal cost estimate

    getattr(st, level)(f"**Recommended Action:** {action}")
    st.write(f"📝 {reason}")
    c1, c2 = st.columns(2)
    c1.metric("Surplus / Shortfall", f"{surplus} meals")
    c2.metric("Potential Savings if Followed", f"₹{savings}")
else:
    st.info("Calculate demand first to unlock recommendations.")

st.divider()

st.subheader("📊 Historical Sustainability Logs")
chart_tab1, chart_tab2 = st.tabs(["📈 Attendance vs Portions Sold", "🌱 Food Waste History"])

with chart_tab1:
    st.markdown("**Historical Attendance vs. Food Demands (Past 30 Days)**")
    recent_df = df.tail(30).copy()
    st.line_chart(recent_df, x="date", y=["attendance_count", "portions_sold"], height=300)

with chart_tab2:
    st.markdown("**Historical Daily Food Waste Log (kg)**")
    st.bar_chart(recent_df, x="date", y="food_wasted_kg", height=300)

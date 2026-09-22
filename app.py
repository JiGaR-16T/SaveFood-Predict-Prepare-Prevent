import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SaveFood - Canteen Intelligence", page_icon="🥗", layout="wide")

st.markdown("""
    <style>
    /* Premium Dark Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0c1420 100%);
        background-attachment: fixed;
    }

    /* Force readable text everywhere by default */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #e2e8f0 !important;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #0369a1 50%, #075985 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.25);
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .hero-card h1, .hero-card p, .hero-card b { color: #f0f9ff !important; }

    /* Section headers (Input Params, Decision Analytics, etc.) */
    .stApp h2, .stApp h3 {
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }

    /* Glass panels for inputs/metrics */
    .stMetric, div[data-testid="stVerticalBlock"] > div:has(> div.stNumberInput) {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        padding: 15px !important;
        border-radius: 15px !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .stMetric label, .stMetric div { color: #e2e8f0 !important; }

    /* Inputs, selects, number fields */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.35);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #075985 100%) !important;
        transform: translateY(-2px);
    }

    /* Success / Warning / Error / Info boxes */
    div[data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 12px !important;
        border-left: 4px solid #38bdf8 !important;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] b {
        color: #f0f9ff !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #94a3b8 !important; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; font-weight: 700 !important; }

    /* Divider */
    hr { border-color: #334155 !important; }
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
        <h1 style="color: #f0f9ff; margin: 0; font-size: 2.6rem; font-weight: 700;">🥗 SaveFood: Demand Intelligence</h1>
        <p style="font-size: 1.15rem; color: #f0f9ff; margin-top: 10px; letter-spacing: 0.5px;">
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
    if surplus >= 0:
        c1.metric("Surplus", f"{surplus} meals", delta="Over-prepared")
    else:
        c1.metric("Shortfall", f"{abs(surplus)} meals", delta="Under-prepared", delta_color="inverse")
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

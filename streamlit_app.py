import os
import pickle

import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")

st.set_page_config(page_title="EVRange Predictor", page_icon="🔋", layout="centered")


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


model = load_model()

st.title("🔋 EVRange: Real-World EV Range Predictor")
st.caption("Enter the vehicle specs and driving conditions to estimate real-world range (km).")

if model is None:
    st.warning(
        "No trained model found at `model/best_model.pkl` yet. "
        "This UI is ready — plug in the exported model from the Kaggle notebook to enable predictions."
    )

st.header("Vehicle Specs")
col1, col2 = st.columns(2)
with col1:
    make = st.selectbox(
        "Make",
        ["Tata", "Mahindra", "MG", "Hyundai", "Kia", "BYD", "BMW", "Mercedes",
         "Audi", "Volvo", "Porsche", "Jaguar", "Citroen", "Maruti Suzuki", "Toyota"],
    )
    battery_capacity_kWh = st.number_input("Current battery capacity (kWh)", min_value=0.0, value=35.3, step=0.1)
    battery_capacity_kWh_new = st.number_input("Original battery capacity (kWh)", min_value=0.0, value=40.5, step=0.1)
    motor_power_kW = st.number_input("Motor power (kW)", min_value=0.0, value=105.0, step=1.0)
    vehicle_weight_kg = st.number_input("Vehicle weight (kg)", min_value=0.0, value=1610.0, step=10.0)
    price_lakh_INR = st.number_input("Price (INR lakh)", min_value=0.0, value=17.5, step=0.5)
with col2:
    drag_coefficient_Cd = st.number_input("Drag coefficient (Cd)", min_value=0.0, value=0.34, step=0.01)
    frontal_area_m2 = st.number_input("Frontal area (m²)", min_value=0.0, value=2.4, step=0.1)
    rolling_resistance_Cr = st.number_input("Rolling resistance (Cr)", min_value=0.0, value=0.0122, step=0.001, format="%.4f")
    motor_efficiency_pct = st.number_input("Motor efficiency (%)", min_value=0.0, max_value=100.0, value=91.0, step=1.0)
    battery_age_years = st.number_input("Battery age (years)", min_value=0.0, value=1.5, step=0.1)

st.header("Driving & Environmental Conditions")
col3, col4 = st.columns(2)
with col3:
    avg_driving_speed_kmh = st.number_input("Average driving speed (km/h)", min_value=0.0, value=30.0, step=1.0)
    ambient_temperature_C = st.number_input("Ambient temperature (°C)", value=30.0, step=1.0)
    ac_usage = st.selectbox("AC usage", options=[0, 1], format_func=lambda x: "On" if x == 1 else "Off")
    passengers = st.number_input("Passengers", min_value=1, max_value=8, value=1, step=1)
with col4:
    terrain_type = st.selectbox("Terrain type", options=[0, 1, 2], format_func=lambda x: ["Flat", "Hilly", "Mixed"][x])
    traffic_condition = st.selectbox("Traffic condition", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
    driving_style = st.selectbox("Driving style", options=[0, 1, 2], format_func=lambda x: ["Calm", "Normal", "Aggressive"][x])
    tyre_pressure_deviation_psi = st.number_input("Tyre pressure deviation (psi)", value=0, step=1)

city_road_type = st.selectbox(
    "City / road type",
    ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune", "Hyderabad", "Kolkata",
     "Highway_NH", "Expressway", "Hill_Station", "Rural_Road", "Ghat_Road"],
)

if st.button("Predict Range", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        "make": make,
        "battery_capacity_kWh": battery_capacity_kWh,
        "battery_capacity_kWh_new": battery_capacity_kWh_new,
        "motor_power_kW": motor_power_kW,
        "vehicle_weight_kg": vehicle_weight_kg,
        "drag_coefficient_Cd": drag_coefficient_Cd,
        "frontal_area_m2": frontal_area_m2,
        "rolling_resistance_Cr": rolling_resistance_Cr,
        "motor_efficiency_pct": motor_efficiency_pct,
        "avg_driving_speed_kmh": avg_driving_speed_kmh,
        "ambient_temperature_C": ambient_temperature_C,
        "ac_usage": ac_usage,
        "terrain_type": terrain_type,
        "traffic_condition": traffic_condition,
        "driving_style": driving_style,
        "passengers": passengers,
        "battery_age_years": battery_age_years,
        "tyre_pressure_deviation_psi": tyre_pressure_deviation_psi,
        "city_road_type": city_road_type,
        "price_lakh_INR": price_lakh_INR,
    }])

    if model is None:
        st.error("Cannot predict yet — no trained model found at `model/best_model.pkl`.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted real-world range: **{prediction:.1f} km**")

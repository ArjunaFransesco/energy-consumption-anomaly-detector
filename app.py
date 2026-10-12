import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Smart Meter Energy Consumption Anomaly Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Smart Meter Energy Consumption Anomaly Detector")
st.markdown("**Domain**: `Energy & Utilities / Anomaly Detection` | **Tech Stack**: `Isolation Forest, LightGBM, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    ambient_temperature_c = st.slider("Ambient Temperature C", float(-5.0), float(42.0), float(22.0))
    relative_humidity_pct = st.slider("Relative Humidity Pct", float(15.0), float(98.0), float(62.0))
    rolling_24h_peak_load_kw = st.slider("Rolling 24H Peak Load Kw", float(10.0), float(120.0), float(48.0))
    solar_irradiance_wm2 = st.slider("Solar Irradiance Wm2", float(0.0), float(1100.0), float(450.0))
    time_of_day_cyclical_sin = st.slider("Time Of Day Cyclical Sin", float(-1.0), float(1.0), float(0.0))
    industrial_consumer_flag = st.slider("Industrial Consumer Flag", int(0), int(1), int(0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"ambient_temperature_c": ambient_temperature_c, "relative_humidity_pct": relative_humidity_pct, "rolling_24h_peak_load_kw": rolling_24h_peak_load_kw, "solar_irradiance_wm2": solar_irradiance_wm2, "time_of_day_cyclical_sin": time_of_day_cyclical_sin, "industrial_consumer_flag": industrial_consumer_flag}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `is_anomaly_event`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)

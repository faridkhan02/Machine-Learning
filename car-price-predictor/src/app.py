"""
app.py
Streamlit front-end for the Car Price Predictor — dashboard/instrument-
cluster styled UI.

Run with:
    streamlit run app.py
"""

import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import MODEL_PATH, train
from data_loader import load_data

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# ----------------------------------------------------------------------------
# THEME — instrument-cluster / dashboard styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg: #14161A;
    --panel: #1C1F26;
    --panel-border: #2B2F38;
    --chrome: #E8E6E1;
    --chrome-dim: #9A9EA8;
    --amber: #F2A93B;
    --amber-dim: #7A5A22;
    --teal: #3E8E8C;
    --danger: #D64545;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--chrome);
}

.stApp {
    background:
        radial-gradient(circle at 50% 0%, #1a1d24 0%, var(--bg) 55%);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 760px;
}

/* ---- Hero / gauge header ---- */
.dash-hero {
    text-align: center;
    padding: 1.6rem 1rem 2rem 1rem;
    border-bottom: 1px solid var(--panel-border);
    margin-bottom: 2rem;
}
.dash-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.25em;
    font-size: 0.7rem;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.dash-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: var(--chrome);
    margin: 0;
    letter-spacing: -0.01em;
}
.dash-sub {
    font-family: 'Inter', sans-serif;
    color: var(--chrome-dim);
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* ---- Panel card ---- */
.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.4rem;
}
.panel-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--chrome-dim);
    margin-bottom: 0.8rem;
}

/* ---- Form widget overrides ---- */
div[data-testid="stForm"] {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 1.4rem 1.5rem 0.6rem 1.5rem;
}
.stSelectbox label, .stNumberInput label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--chrome-dim) !important;
}
div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: #14161A !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--chrome) !important;
    border-radius: 6px !important;
}
.stNumberInput input {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---- Buttons ---- */
.stButton button, .stFormSubmitButton button {
    background: linear-gradient(180deg, var(--amber) 0%, #d99226 100%) !important;
    color: #14161A !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 12px rgba(242, 169, 59, 0.25);
    width: 100%;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    box-shadow: 0 2px 18px rgba(242, 169, 59, 0.45);
}

/* ---- Odometer readout ---- */
.readout {
    background: #0E1013;
    border: 1px solid var(--amber-dim);
    border-radius: 10px;
    padding: 1.6rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.readout::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
}
.readout-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--chrome-dim);
    margin-bottom: 0.5rem;
}
.readout-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 3rem;
    color: var(--amber);
    letter-spacing: 0.03em;
    text-shadow: 0 0 18px rgba(242, 169, 59, 0.35);
}
.readout-sub {
    font-family: 'Inter', sans-serif;
    color: var(--chrome-dim);
    font-size: 0.85rem;
    margin-top: 0.6rem;
}

/* ---- Gauge / range bar ---- */
.gauge-wrap {
    margin-top: 1.3rem;
}
.gauge-track {
    position: relative;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--teal), var(--amber), var(--danger));
    margin: 0.4rem 0 0.3rem 0;
}
.gauge-needle {
    position: absolute;
    top: -5px;
    width: 3px;
    height: 18px;
    background: var(--chrome);
    box-shadow: 0 0 6px rgba(232,230,225,0.8);
    border-radius: 2px;
    transform: translateX(-50%);
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--chrome-dim);
}

hr {
    border-color: var(--panel-border) !important;
}
.footnote {
    font-family: 'Inter', sans-serif;
    color: var(--chrome-dim);
    font-size: 0.78rem;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="dash-hero">
    <div class="dash-eyebrow">Instrument Cluster · Valuation</div>
    <div class="dash-title">🚗 Car Price Predictor</div>
    <div class="dash-sub">Random Forest valuation, read like an odometer.</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def get_make_model_options():
    df = load_data()
    makes = sorted(df["make"].unique().tolist())
    make_to_models = {
        make: sorted(df[df["make"] == make]["model"].unique().tolist())
        for make in makes
    }
    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    return makes, make_to_models, price_min, price_max


saved = load_model()

if saved is None:
    st.markdown('<div class="panel"><div class="panel-label">Status</div>No trained model on file yet.</div>', unsafe_allow_html=True)
    if st.button("⚙️  Train model now"):
        with st.spinner("Training model..."):
            train()
        st.success("Model trained. Reloading...")
        st.cache_resource.clear()
        st.rerun()
else:
    model = saved["model"]
    feature_columns = saved["feature_columns"]
    makes, make_to_models, price_min, price_max = get_make_model_options()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            make = st.selectbox("Make", makes)
            year = st.number_input("Year", min_value=1990, max_value=2026, value=2020, step=1)

        with col2:
            model_options = make_to_models.get(make, [])
            model_name = st.selectbox("Model", model_options)
            mileage = st.number_input("Mileage (mi)", min_value=0, value=30000, step=1000)

        submitted = st.form_submit_button("PREDICT PRICE")

    if submitted:
        row = pd.DataFrame([{
            "year": year,
            "mileage": mileage,
            f"make_{make}": 1,
            f"model_{model_name}": 1,
        }])
        row = row.reindex(columns=feature_columns, fill_value=0)
        prediction = model.predict(row)[0]

        # Position on the gauge (clamped 0-100%)
        span = max(price_max - price_min, 1)
        pct = max(0, min(100, (prediction - price_min) / span * 100))

        st.markdown(f"""
        <div class="readout">
            <div class="readout-label">Estimated Value</div>
            <div class="readout-value">${prediction:,.0f}</div>
            <div class="readout-sub">{year} {make} {model_name} · {mileage:,} mi</div>
            <div class="gauge-wrap">
                <div class="gauge-track">
                    <div class="gauge-needle" style="left:{pct}%;"></div>
                </div>
                <div class="gauge-labels">
                    <span>${price_min:,}</span>
                    <span>TRAINING RANGE</span>
                    <span>${price_max:,}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔁  Retrain model"):
        with st.spinner("Retraining model..."):
            train()
        st.cache_resource.clear()
        st.success("Model retrained.")
        st.rerun()

st.markdown("""
<div class="footnote">
    Sample dataset is 10 rows — illustrative only. Swap in a real CSV
    via data_loader.py for meaningful accuracy.
</div>
""", unsafe_allow_html=True)

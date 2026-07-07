import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import random

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# THEME / CSS  (blueprint + appraisal-document look)
# ----------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root{
  --ink:#13293D; --ink-soft:#5C7080;
  --blueprint:#2C6E9B; --blueprint-deep:#1B4C6E; --blueprint-light:#E4ECF2;
  --paper:#F1ECDF; --brass:#AE7E32; --brass-deep:#8C641F;
  --success:#3F7D58; --danger:#A6432B; --line:rgba(44,110,155,0.16);
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif; }

.stApp{
  background:
    repeating-linear-gradient(0deg, var(--line) 0px, var(--line) 1px, transparent 1px, transparent 28px),
    repeating-linear-gradient(90deg, var(--line) 0px, var(--line) 1px, transparent 1px, transparent 28px),
    var(--paper);
}

/* header */
.hp-eyebrow{
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:.18em;
  color:var(--blueprint-deep); text-transform:uppercase; margin-bottom:6px;
  display:flex; align-items:center; gap:8px;
}
.hp-eyebrow::before{content:"";width:8px;height:8px;background:var(--brass);display:inline-block;transform:rotate(45deg);}
.hp-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:34px;letter-spacing:-.01em;margin:0 0 4px;color:var(--ink);}
.hp-title span{color:var(--blueprint);}
.hp-header-rule{border-bottom:2px solid var(--ink); padding-bottom:14px; margin-bottom:18px;}
.hp-header-meta{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--ink-soft);text-align:right;line-height:1.6;}

/* section labels */
.hp-section-label{
  font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--blueprint-deep); font-weight:600;
  margin:18px 0 6px; padding-bottom:6px; border-bottom:1px dashed var(--line);
  display:flex; justify-content:space-between;
}
.hp-section-label b{color:var(--ink-soft);font-weight:500;letter-spacing:0;text-transform:none;}

/* card wrapper */
.hp-card{
  background:#FBF9F3; border:1px solid var(--ink); padding:20px 22px 8px; position:relative; margin-bottom:16px;
}
.hp-card::before,.hp-card::after{content:"";position:absolute;width:12px;height:12px;border:2px solid var(--blueprint-deep);}
.hp-card::before{top:-1px;left:-1px;border-right:none;border-bottom:none;}
.hp-card::after{top:-1px;right:-1px;border-left:none;border-bottom:none;}

/* stamp */
.hp-stamp{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  width:172px; height:172px; border:3px solid var(--brass); border-radius:50%;
  transform:rotate(-4deg); margin:6px auto 10px; position:relative;
}
.hp-stamp::before{content:"";position:absolute;inset:6px;border:1px solid var(--brass);border-radius:50%;}
.hp-stamp .lbl{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.16em;color:var(--brass-deep);text-transform:uppercase;}
.hp-stamp .amt{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:24px;color:var(--ink);line-height:1.15;margin:4px 0 2px;text-align:center;}

.hp-conf-chip{
  display:inline-block; margin:4px auto 0; padding:4px 12px; font-family:'IBM Plex Mono',monospace;
  font-size:11px; letter-spacing:.08em; text-transform:uppercase; border:1px solid var(--success); color:var(--success);
}
.hp-conf-chip.medium{border-color:var(--brass-deep);color:var(--brass-deep);}
.hp-conf-chip.low{border-color:var(--danger);color:var(--danger);}
.hp-chip-center{text-align:center;}

.hp-summary-row{
  display:flex; justify-content:space-between; padding:6px 0;
  border-bottom:1px dotted var(--line); font-family:'IBM Plex Mono',monospace; font-size:12.5px;
}
.hp-summary-row span:first-child{color:var(--ink-soft);}
.hp-summary-row span:last-child{font-weight:600;color:var(--ink);}

.hp-footnote{
  font-family:'IBM Plex Mono',monospace; font-size:10.5px; color:var(--ink-soft);
  text-align:center; letter-spacing:.02em; margin-top:20px;
}

/* streamlit widget restyling */
div[data-testid="stNumberInput"] input, div[data-baseweb="input"] input, div[data-baseweb="select"] > div{
  font-family:'IBM Plex Mono',monospace !important; border:1px solid #C9BFA3 !important; border-radius:0 !important;
  background:#fff !important; color:var(--ink) !important;
}
div[data-testid="stSlider"] > div > div > div > div{ background:var(--blueprint-deep) !important; }
.stButton>button{
  background:var(--blueprint-deep); color:#fff; border:none; border-radius:0; padding:12px 18px;
  font-family:'Space Grotesk',sans-serif; font-weight:600; letter-spacing:.03em; text-transform:uppercase;
  width:100%;
}
.stButton>button:hover{background:var(--ink); color:#fff;}
.stDownloadButton>button{
  background:transparent; border:1.5px solid var(--ink) !important; color:var(--ink); border-radius:0;
  font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:.06em; text-transform:uppercase; width:100%;
}
.stDownloadButton>button:hover{background:var(--ink); color:#fff;}
label, .stMarkdown p{ color:var(--ink); }
hr{border-color:var(--line);}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
if "file_no" not in st.session_state:
    st.session_state.file_no = random.randint(1000, 9999)

hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown("""
        <div class="hp-eyebrow">Automated Valuation Model &middot; Rev. 03</div>
        <div class="hp-title">House Price <span>Predictor</span></div>
    """, unsafe_allow_html=True)
with hc2:
    st.markdown(f"""
        <div class="hp-header-meta">
            FILE NO. HPX-{st.session_state.file_no}<br>
            MODEL: GRADIENT BOOST v2.4
        </div>
    """, unsafe_allow_html=True)
st.markdown('<div class="hp-header-rule"></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# STATIC FEATURE IMPORTANCE (global model property, not per-prediction)
# ----------------------------------------------------------------------------
IMPORTANCE = [
    ("Construction Grade", 27),
    ("Living Area", 22),
    ("Latitude", 12),
    ("Nearby Living Area", 9),
    ("Longitude", 8),
    ("Waterfront", 7),
    ("View Quality", 6),
    ("Year Built", 5),
    ("Bathrooms", 4),
]

# ----------------------------------------------------------------------------
# LAYOUT
# ----------------------------------------------------------------------------
left, right = st.columns([1.35, 1], gap="medium")

with left:
    st.markdown('<div class="hp-card">', unsafe_allow_html=True)

    st.markdown('<p class="hp-section-label">Structure <b>rooms &amp; footprint</b></p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=15, value=3)
    bathrooms = c2.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.5, step=0.25)
    floors = c3.number_input("Floors", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    year_built = c4.number_input("Year Built", min_value=1880, max_value=2026, value=2002)

    c5, c6 = st.columns(2)
    living_area = c5.number_input("Living Area (sqft)", min_value=0, value=1800)
    lot_area = c6.number_input("Lot Area (sqft)", min_value=0, value=5200)

    c7, c8 = st.columns(2)
    above_area = c7.number_input("Above-Ground Area (sqft)", min_value=0, value=1400)
    basement_area = c8.number_input("Basement Area (sqft)", min_value=0, value=400)

    st.markdown('<p class="hp-section-label">Condition &amp; Grade <b>quality inputs</b></p>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    condition = c9.slider("Condition (1-5)", 1, 5, 3)
    view = c10.slider("View Quality (0-4)", 0, 4, 0)
    grade = st.slider("Construction Grade (1-13)", 1, 13, 8)

    c11, c12 = st.columns(2)
    waterfront = c11.radio("Waterfront", ["No", "Yes"], horizontal=True) == "Yes"
    year_renovated = c12.number_input("Year Renovated (0 = none)", min_value=0, max_value=2026, value=0)

    st.markdown('<p class="hp-section-label">Lot &amp; Location <b>geography</b></p>', unsafe_allow_html=True)
    c13, c14, c15 = st.columns(3)
    zipcode = c13.text_input("Zipcode", value="98115", max_chars=5)
    lat = c14.number_input("Latitude", value=47.6870, step=0.0001, format="%.4f")
    lon = c15.number_input("Longitude", value=-122.3120, step=0.0001, format="%.4f")

    c16, c17 = st.columns(2)
    nearby_living = c16.number_input("Nearby Living Area (sqft)", min_value=0, value=1750)
    nearby_lot = c17.number_input("Nearby Lot Area (sqft)", min_value=0, value=5400)

    predict_clicked = st.button("▸ Predict House Price")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PREDICTION LOGIC  (simulated valuation model)
# ----------------------------------------------------------------------------
def predict_price(bedrooms, bathrooms, floors, living_area, lot_area, above_area,
                   basement_area, year_built, year_renovated, condition, view,
                   grade, waterfront, lat, nearby_living):
    grade_factor = 1.13 ** (grade - 7)
    condition_factor = 1 + (condition - 3) * 0.035
    price_per_sqft = 165 + (grade - 7) * 12

    living_value = living_area * price_per_sqft * grade_factor * condition_factor
    basement_value = basement_area * price_per_sqft * 0.55
    bed_bath_bonus = bedrooms * 6500 + bathrooms * 11000 + floors * 8000
    waterfront_bonus = (150000 + view * 45000) if waterfront else view * 22000
    age = max(0, 2026 - (year_built or 2026))
    age_penalty = min(age * 750, 90000)
    renovation_bonus = 28000 if year_renovated > 0 else 0
    lot_bonus = min(lot_area * 1.8, 55000)
    nearby_adj = ((nearby_living or living_area) - living_area) * price_per_sqft * 0.15
    location_factor = 1 + max(-0.08, min(0.12, (lat - 47.55) * 0.35)) if lat else 1

    price = (42000 + living_value + basement_value + bed_bath_bonus + waterfront_bonus
             + lot_bonus + renovation_bonus + nearby_adj - age_penalty) * location_factor
    price = max(85000, round(price / 500) * 500)
    return price


if "last_prediction" not in st.session_state or predict_clicked:
    price = predict_price(bedrooms, bathrooms, floors, living_area, lot_area, above_area,
                           basement_area, year_built, year_renovated, condition, view,
                           grade, waterfront, lat, nearby_living)

    required_filled = sum(1 for v in [living_area, lot_area, above_area + basement_area,
                                       year_built, lat] if v and v > 0)
    if required_filled >= 5:
        confidence, conf_class = "High", ""
    elif required_filled >= 3:
        confidence, conf_class = "Medium", "medium"
    else:
        confidence, conf_class = "Low", "low"

    st.session_state.last_prediction = {
        "price": price, "confidence": confidence, "conf_class": conf_class,
        "bedrooms": bedrooms, "bathrooms": bathrooms, "floors": floors,
        "living_area": living_area, "lot_area": lot_area, "above_area": above_area,
        "basement_area": basement_area, "year_built": year_built,
        "year_renovated": year_renovated, "condition": condition, "view": view,
        "grade": grade, "waterfront": waterfront, "zipcode": zipcode,
        "latitude": lat, "longitude": lon, "nearby_living_area": nearby_living,
        "nearby_lot_area": nearby_lot,
    }

pred = st.session_state.last_prediction

# ----------------------------------------------------------------------------
# RIGHT COLUMN: RESULTS
# ----------------------------------------------------------------------------
with right:
    st.markdown('<div class="hp-card">', unsafe_allow_html=True)
    st.markdown('<p class="hp-section-label">Estimated Price <b>valuation output</b></p>', unsafe_allow_html=True)

    per_sqft = int(pred["price"] / pred["living_area"]) if pred["living_area"] else 0
    st.markdown(f"""
        <div class="hp-stamp">
            <div class="lbl">Appraised at</div>
            <div class="amt">$ {pred['price']:,.0f}</div>
            <div class="lbl">{f"${per_sqft:,} / sqft" if per_sqft else "&mdash;"}</div>
        </div>
        <div class="hp-chip-center">
            <span class="hp-conf-chip {pred['conf_class']}">Confidence: {pred['confidence']}</span>
        </div>
    """, unsafe_allow_html=True)

    # ---- Gauge ----
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred["price"],
        number={"prefix": "$", "valueformat": ",.0f", "font": {"family": "IBM Plex Mono", "size": 22, "color": "#13293D"}},
        gauge={
            "axis": {"range": [200000, 1250000], "tickformat": "$.2s", "tickfont": {"family": "IBM Plex Mono", "size": 10}},
            "bar": {"color": "#13293D", "thickness": 0.25},
            "bgcolor": "#E4DDC6",
            "borderwidth": 0,
            "steps": [
                {"range": [200000, 550000], "color": "#3F7D58"},
                {"range": [550000, 900000], "color": "#2C6E9B"},
                {"range": [900000, 1250000], "color": "#AE7E32"},
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10),
                             paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"})
    st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # ---- Feature importance ----
    st.markdown('<p class="hp-section-label">Feature Importance <b>global model</b></p>', unsafe_allow_html=True)
    names = [f[0] for f in IMPORTANCE][::-1]
    values = [f[1] for f in IMPORTANCE][::-1]
    fig_bar = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color="#2C6E9B"),
        text=[f"{v}%" for v in values], textposition="outside",
        textfont={"family": "IBM Plex Mono", "size": 11, "color": "#13293D"},
    ))
    fig_bar.update_layout(
        height=280, margin=dict(l=10, r=30, t=5, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, 32]),
        yaxis=dict(tickfont={"family": "IBM Plex Mono", "size": 11, "color": "#5C7080"}),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # ---- Property summary ----
    st.markdown('<p class="hp-section-label">Property Summary <b>at prediction</b></p>', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="hp-summary-row"><span>Bedrooms</span><span>{pred['bedrooms']}</span></div>
        <div class="hp-summary-row"><span>Bathrooms</span><span>{pred['bathrooms']}</span></div>
        <div class="hp-summary-row"><span>Grade</span><span>{pred['grade']}</span></div>
        <div class="hp-summary-row"><span>Living Area</span><span>{pred['living_area']:,} sqft</span></div>
        <div class="hp-summary-row"><span>Built Year</span><span>{pred['year_built']}</span></div>
    """, unsafe_allow_html=True)

    # ---- Download ----
    report = "\n".join([
        "HOUSE PRICE PREDICTION REPORT",
        "================================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Estimated Price: ${pred['price']:,.0f}",
        f"Confidence: {pred['confidence']}",
        "",
        "PROPERTY DETAILS",
        "--------------------------------",
        f"Bedrooms: {pred['bedrooms']}",
        f"Bathrooms: {pred['bathrooms']}",
        f"Floors: {pred['floors']}",
        f"Living Area: {pred['living_area']} sqft",
        f"Lot Area: {pred['lot_area']} sqft",
        f"Above Area: {pred['above_area']} sqft",
        f"Basement Area: {pred['basement_area']} sqft",
        f"Year Built: {pred['year_built']}",
        f"Year Renovated: {pred['year_renovated'] or 'None'}",
        f"Condition: {pred['condition']}/5",
        f"View: {pred['view']}/4",
        f"Grade: {pred['grade']}/13",
        f"Waterfront: {'Yes' if pred['waterfront'] else 'No'}",
        f"Zipcode: {pred['zipcode']}",
        f"Latitude: {pred['latitude']}",
        f"Longitude: {pred['longitude']}",
        f"Nearby Living Area: {pred['nearby_living_area']} sqft",
        f"Nearby Lot Area: {pred['nearby_lot_area']} sqft",
    ])
    st.download_button("⬇ Download Prediction", data=report,
                        file_name="house_price_prediction.txt", mime="text/plain")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <p class="hp-footnote">Estimate generated from a simulated valuation model for demonstration purposes &middot; not a substitute for a licensed appraisal</p>
""", unsafe_allow_html=True)
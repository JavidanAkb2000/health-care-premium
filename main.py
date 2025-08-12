# app.py
# Streamlit frontend for Health-Care Premium ML model
# Run:  streamlit run app.py

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from prediction_helper import predict  # your helper

# -----------------------------
# Page + Theme
# -----------------------------
st.set_page_config(
    page_title="Health-Care Premium Estimator",
    page_icon="💸",
    layout="wide",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Simple Streamlit frontend for health-care premium prediction."
    },
)

HARD_CATEGORIES = {
    "gender": ["Male", "Female"],
    "region": ["Northeast", "Northwest", "Southeast", "Southwest"],
    "marital_status": ["Unmarried", "Married"],
    "bmi_category": ["Overweight", "Underweight", "Normal", "Obesity"],
    # given messy values -> we'll normalize to: Regular, Occasional, Not Smoking
    "smoking_status_raw": [
        "Regular", "No Smoking", "Occasional", "Smoking=0", "Does Not Smoke", "Not Smoking"
    ],
    "employment_status": ["Self-Employed", "Freelancer", "Salaried"],
    "income_level": ["> 40L", "<10L", "10L - 25L", "25L - 40L"],
    "medical_history": [
        "High blood pressure", "No Disease", "Diabetes & High blood pressure",
        "Diabetes & Heart disease", "Diabetes", "Diabetes & Thyroid",
        "Heart disease", "Thyroid", "High blood pressure & Heart disease"
    ],
    "insurance_plan": ["Silver", "Bronze", "Gold"],
}

# Unified smoking values you can align your model to
SMOKING_NORMALIZATION = {
    "Regular": "Regular",
    "Occasional": "Occasional",
    "No Smoking": "Not Smoking",
    "Does Not Smoke": "Not Smoking",
    "Not Smoking": "Not Smoking",
    "Smoking=0": "Not Smoking",
}

# Optional income midpoint engineering (L = lakh)
INCOME_MIDPOINT_L = {
    "<10L": 5,
    "10L - 25L": 17.5,
    "25L - 40L": 32.5,
    "> 40L": 45,  # conservative placeholder; adjust to your cap
}

# Pre-select reasonable defaults
DEFAULTS = {
    "gender": "Male",
    "region": "Northeast",
    "marital_status": "Unmarried",
    "bmi_category": "Normal",
    "smoking_status_raw": "Not Smoking",
    "employment_status": "Salaried",
    "income_level": "10L - 25L",
    "medical_history": "No Disease",
    "insurance_plan": "Silver",
    "age": 35,
    "bmi": 24.5,
    "dependents": 0,
    "genetical_risk": 0,  # NEW default
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def example_payload():
    return {
        "gender": DEFAULTS["gender"],
        "region": DEFAULTS["region"],
        "marital_status": DEFAULTS["marital_status"],
        "bmi_category": DEFAULTS["bmi_category"],
        "smoking_status": "Not Smoking",
        "employment_status": DEFAULTS["employment_status"],
        "income_level": DEFAULTS["income_level"],
        "income_midpoint_lakh": INCOME_MIDPOINT_L[DEFAULTS["income_level"]],
        "medical_history": DEFAULTS["medical_history"],
        "insurance_plan": DEFAULTS["insurance_plan"],
        "age": DEFAULTS["age"],
        "bmi": DEFAULTS["bmi"],
        "dependents": DEFAULTS["dependents"],
        "income_in_lakhs": INCOME_MIDPOINT_L[DEFAULTS["income_level"]],  # NEW
        "genetical_risk": DEFAULTS["genetical_risk"],  # NEW
    }

def normalize_inputs(raw: dict) -> dict:
    smoking_norm = SMOKING_NORMALIZATION.get(raw["smoking_status_raw"], "Not Smoking")
    income_mid = INCOME_MIDPOINT_L.get(raw["income_level"], np.nan)

    return {
        "gender": raw["gender"],
        "region": raw["region"],
        "marital_status": raw["marital_status"],
        "bmi_category": raw["bmi_category"],
        "smoking_status": smoking_norm,
        "employment_status": raw["employment_status"],
        "income_level": raw["income_level"],
        "income_midpoint_lakh": income_mid,
        "medical_history": raw["medical_history"],
        "insurance_plan": raw["insurance_plan"],
        "age": raw["age"],
        "bmi": raw["bmi"],
        "dependents": raw["dependents"],
        # Pass-throughs
        "income_in_lakhs": raw.get("income_in_lakhs", income_mid),  # prefer explicit numeric
        "genetical_risk": raw.get("genetical_risk", 0),
    }

@st.cache_resource(show_spinner=False)
def load_model(uploaded_file):
    # Optional fallback path: sklearn Pipeline/estimator via pickle
    import pickle
    return pickle.load(uploaded_file)

def predict_with_model(model, df: pd.DataFrame):
    res = {}
    if hasattr(model, "predict"):
        try:
            pred = model.predict(df)[0]
            res["premium"] = float(pred)
        except Exception as e:
            res["error"] = f"Model predict failed: {e}"
    return res

def rupee_format(x):
    try:
        return f"₹{x:,.0f}"
    except Exception:
        return str(x)

# -----------------------------
# Sidebar: Model & Options
# -----------------------------
with st.sidebar:
    st.header("🔧 Model & Options")

    model_file = st.file_uploader(
        "Upload pickled model/pipeline (.pkl)", type=["pkl", "pickle"],
        help="Upload a pickled sklearn Pipeline or estimator that accepts the features displayed below."
    )
    show_engineered = st.toggle("Show engineered feature(s)", value=True)
    show_payload = st.toggle("Show request JSON", value=True)
    show_df = st.toggle("Show model dataframe row", value=False)

    st.caption("Tip: Save a Pipeline that includes your encoder(s) so this UI can feed raw categories cleanly.")

# -----------------------------
# Header
# -----------------------------
st.title("💸 Health-Care Premium Estimator")
st.caption("No-nonsense UI for your premium model.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("premium_form", clear_on_submit=False):
    st.subheader("Member details")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", HARD_CATEGORIES["gender"], index=HARD_CATEGORIES["gender"].index(DEFAULTS["gender"]))
        marital_status = st.selectbox("Marital Status", HARD_CATEGORIES["marital_status"], index=HARD_CATEGORIES["marital_status"].index(DEFAULTS["marital_status"]))
        employment_status = st.selectbox("Employment Status", HARD_CATEGORIES["employment_status"], index=HARD_CATEGORIES["employment_status"].index(DEFAULTS["employment_status"]))
    with c2:
        region = st.selectbox("Region", HARD_CATEGORIES["region"], index=HARD_CATEGORIES["region"].index(DEFAULTS["region"]))
        bmi_category = st.selectbox("BMI Category", HARD_CATEGORIES["bmi_category"], index=HARD_CATEGORIES["bmi_category"].index(DEFAULTS["bmi_category"]))
        insurance_plan = st.selectbox("Insurance Plan", HARD_CATEGORIES["insurance_plan"], index=HARD_CATEGORIES["insurance_plan"].index(DEFAULTS["insurance_plan"]))
    with c3:
        smoking_status_raw = st.selectbox("Smoking Status (as given)", HARD_CATEGORIES["smoking_status_raw"], index=HARD_CATEGORIES["smoking_status_raw"].index(DEFAULTS["smoking_status_raw"]))
        income_level = st.selectbox("Income Level", HARD_CATEGORIES["income_level"], index=HARD_CATEGORIES["income_level"].index(DEFAULTS["income_level"]))
        medical_history = st.selectbox("Medical History", HARD_CATEGORIES["medical_history"], index=HARD_CATEGORIES["medical_history"].index(DEFAULTS["medical_history"]))
    with c4:
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=DEFAULTS["age"], step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(DEFAULTS["bmi"]), step=0.1, format="%.1f")
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=int(DEFAULTS["dependents"]), step=1)

    st.divider()

    # NEW: direct numeric inputs for income_in_lakhs and genetical_risk
    c5, c6 = st.columns(2)
    with c5:
        income_in_lakhs = st.number_input(
            "Income in Lakhs (numeric)",
            min_value=0.0, max_value=100.0,
            value=float(INCOME_MIDPOINT_L[DEFAULTS["income_level"]]),
            step=0.5, format="%.1f",
            help="Enter your actual income in lakhs. Passed to model as `income_lakhs`."
        )
    with c6:
        genetical_risk = st.number_input(
            "Genetical Risk (integer)",
            min_value=0, max_value=20, value=int(DEFAULTS["genetical_risk"]), step=1,
            help="Integer score like 0,1,2,... indicating genetic predisposition."
        )

    st.divider()

    cA, cB = st.columns([1, 4])
    with cA:
        submit = st.form_submit_button("Predict Premium", use_container_width=True, type="primary")
    with cB:
        reset = st.form_submit_button("Reset to Defaults", use_container_width=True)

# Reset logic
if 'last_reset_ts' not in st.session_state:
    st.session_state['last_reset_ts'] = None

if reset:
    st.session_state['last_reset_ts'] = datetime.utcnow().isoformat()
    st.success("Inputs reset to defaults. Adjust and predict again.")

# -----------------------------
# Build Payload
# -----------------------------
raw_inputs = {
    "gender": gender,
    "region": region,
    "marital_status": marital_status,
    "bmi_category": bmi_category,
    "smoking_status_raw": smoking_status_raw,
    "employment_status": employment_status,
    "income_level": income_level,
    "medical_history": medical_history,
    "insurance_plan": insurance_plan,
    "age": int(age),
    "bmi": float(bmi),
    "dependents": int(dependents),
    "income_in_lakhs": float(income_in_lakhs),    # NEW
    "genetical_risk": int(genetical_risk),        # NEW
}

payload = normalize_inputs(raw_inputs)
payload_df = pd.DataFrame([payload])

# -----------------------------
# Output
# -----------------------------
st.subheader("Prediction")

if submit:
    # Hard age gate
    if payload["age"] < 18:
        st.error("Age must be at least 18 to calculate premium.")
        st.stop()  # prevent any further calculation

    # Validate engineered fields
    if np.isnan(payload["income_midpoint_lakh"]):
        st.warning("Income midpoint is NaN due to unrecognized income_level. Check INCOME_MIDPOINT_L mapping.")

    # Show what is going to the model (regardless of prediction path)
    info_cols = st.columns(3)
    with info_cols[0]:
        st.metric("Smoking (normalized)", payload["smoking_status"])
        st.metric("Income Level", payload["income_level"])
    with info_cols[1]:
        st.metric("Income Midpoint (L)", payload["income_midpoint_lakh"])
        st.metric("BMI", payload["bmi"])
    with info_cols[2]:
        st.metric("Age", payload["age"])
        st.metric("Dependents", payload["dependents"])

    # Show the two new numerics too
    more_cols = st.columns(2)
    with more_cols[0]:
        st.metric("Income in Lakhs (numeric)", payload.get("income_in_lakhs", np.nan))
    with more_cols[1]:
        st.metric("Genetical Risk", payload.get("genetical_risk", 0))

    # --- Primary path: use your prediction_helper.predict(payload) ---
    premium = None
    try:
        premium = predict(payload)  # expects numeric return
    except Exception as e:
        st.error(f"prediction_helper.predict failed: {e}")

    # Optional fallback: uploaded sklearn model (if helper returns None)
    if premium is None:
        model = None
        if model_file is not None:
            try:
                model = load_model(model_file)
                st.caption("✅ Model loaded (fallback).")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

        if model is not None:
            result = predict_with_model(model, payload_df)
            if "error" in result:
                st.error(result["error"])
            else:
                premium = result.get("premium", None)

    # Final display
    if premium is None:
        st.info("No premium returned. Ensure prediction_helper.predict returns a numeric value.")
    else:
        st.success(f"Estimated Premium: {rupee_format(float(premium))}")

    # Optional views
    with st.expander("🔍 Features sent to model / helper", expanded=bool(show_df)):
        st.dataframe(payload_df)

    if show_engineered:
        st.caption("Engineered field(s): income_midpoint_lakh derived from income_level.")

    if show_payload:
        st.code(json.dumps(payload, indent=2), language="json")

# -----------------------------
# Footer / Notes
# -----------------------------
st.divider()
with st.container():
    st.caption(
        "Notes:\n"
        "- Smoking labels are normalized to: Regular / Occasional / Not Smoking.\n"
        "- Income midpoint (in lakhs) is a simple heuristic; align with your training logic.\n"
        "- For best results, upload a sklearn Pipeline (preprocessing + model) so raw categories map correctly."
    )

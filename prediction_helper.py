# prediction_helper.py
# codebasics ML course: codebasics.io, all rights reserved

import pandas as pd
import joblib


# ---- Load artifacts
model_young = joblib.load("artifacts/model_young.joblib")
model_rest = joblib.load("artifacts/model_rest.joblib")
scaler_young = joblib.load("artifacts/scaler_young.joblib")
scaler_rest = joblib.load("artifacts/scaler_rest.joblib")


# ---- Small util
def _get(d, *keys, default=None):
    """Return the first present key in dict d, else default."""
    for k in keys:
        if k in d:
            return d[k]
    return default


# ---- Feature engineering
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0,
    }
    if not isinstance(medical_history, str):
        return 0.0

    # support "A & B" or "A&B"
    diseases = [s.strip().lower() for s in medical_history.replace(" & ", "&").split("&")]
    total_risk_score = sum(risk_scores.get(d, 0) for d in diseases)
    max_score = 14  # 8 (heart) + 6 (diabetes/hbp)
    return total_risk_score / max_score


# ---- Preprocess
def preprocess_input(input_dict):
    """
    Build the exact row expected by the model.
    Accepts both Title Case and lowercase keys.
    Accepts income from:
      - 'Income in Lakhs'
      - 'income_in_lakhs' or 'income_lakhs'
      - 'income_midpoint_lakh'
    Accepts genetical risk from:
      - 'Genetical Risk' or 'genetical_risk' (int-like)
    """
    expected_columns = [
        "age", "number_of_dependants", "income_lakhs", "insurance_plan",
        "genetical_risk", "normalized_risk_score",
        "gender_Male",
        "region_Northwest", "region_Southeast", "region_Southwest",
        "marital_status_Unmarried",
        "bmi_category_Obesity", "bmi_category_Overweight", "bmi_category_Underweight",
        "smoking_status_Occasional", "smoking_status_Regular",
        "employment_status_Salaried", "employment_status_Self-Employed",
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # ----- Numeric basics
    age_val = _get(input_dict, "Age", "age")
    if age_val is not None:
        df.at[0, "age"] = age_val

    dep_val = _get(input_dict, "Number of Dependants", "dependents", "number_of_dependants")
    if dep_val is not None:
        df.at[0, "number_of_dependants"] = dep_val

    # income: map any supported key into income_lakhs
    inc_val = _get(
        input_dict,
        "Income in Lakhs", "income_in_lakhs", "income_lakhs", "income_midpoint_lakh"
    )
    if inc_val is not None:
        df.at[0, "income_lakhs"] = inc_val

    # insurance plan ordinal
    insurance_plan_encoding = {"Bronze": 1, "Silver": 2, "Gold": 3}
    plan_val = _get(input_dict, "Insurance Plan", "insurance_plan")
    if plan_val is not None:
        df.at[0, "insurance_plan"] = insurance_plan_encoding.get(plan_val, 1)

    # genetical risk (optional numeric; coerce to int if possible)
    gr = _get(input_dict, "Genetical Risk", "genetical_risk")
    if gr is not None:
        try:
            df.at[0, "genetical_risk"] = int(gr)
        except Exception:
            # leave at 0 if bad input
            pass

    # ----- One-hots (baseline left at 0)
    gender = _get(input_dict, "Gender", "gender")
    if gender == "Male":
        df.at[0, "gender_Male"] = 1

    region = _get(input_dict, "Region", "region")
    if region == "Northwest":
        df.at[0, "region_Northwest"] = 1
    elif region == "Southeast":
        df.at[0, "region_Southeast"] = 1
    elif region == "Southwest":
        df.at[0, "region_Southwest"] = 1
    # "Northeast" is baseline if that's how you trained

    marital = _get(input_dict, "Marital Status", "marital_status")
    if marital == "Unmarried":
        df.at[0, "marital_status_Unmarried"] = 1

    bmi_cat = _get(input_dict, "BMI Category", "bmi_category")
    if bmi_cat == "Obesity":
        df.at[0, "bmi_category_Obesity"] = 1
    elif bmi_cat == "Overweight":
        df.at[0, "bmi_category_Overweight"] = 1
    elif bmi_cat == "Underweight":
        df.at[0, "bmi_category_Underweight"] = 1
    # "Normal" baseline

    smoking = _get(input_dict, "Smoking Status", "smoking_status")
    if smoking == "Occasional":
        df.at[0, "smoking_status_Occasional"] = 1
    elif smoking == "Regular":
        df.at[0, "smoking_status_Regular"] = 1
    # "Not Smoking" baseline

    emp = _get(input_dict, "Employment Status", "employment_status")
    if emp == "Salaried":
        df.at[0, "employment_status_Salaried"] = 1
    elif emp == "Self-Employed":
        df.at[0, "employment_status_Self-Employed"] = 1
    # "Freelancer" baseline

    # ----- Engineered
    mh = _get(input_dict, "Medical History", "medical_history")
    df.at[0, "normalized_risk_score"] = calculate_normalized_risk(mh)

    # ----- Scale numeric columns
    age_for_scaler = age_val if age_val is not None else 0
    df = handle_scaling(age_for_scaler, df)

    return df


# ---- Scaling
def handle_scaling(age, df):
    # choose scaler
    scaler_object = scaler_young if age <= 25 else scaler_rest

    # tolerate both key names
    cols_to_scale = scaler_object.get("cols_to_scale", scaler_object.get("cols"))
    if cols_to_scale is None:
        raise KeyError("Scaler object missing 'cols_to_scale' or 'cols'.")

    scaler = scaler_object["scaler"]

    # create any missing numeric columns as 0 so transform doesn't crash
    for c in cols_to_scale:
        if c not in df.columns:
            df[c] = 0

    # some scaler artifacts include 'income_level' — satisfy it if needed
    if "income_level" in cols_to_scale and "income_level" not in df.columns:
        df["income_level"] = 0

    # transform in the scaler's column order
    df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale])

    # drop placeholder if we created it
    if "income_level" in df.columns and "income_level" not in [
        "age", "number_of_dependants", "income_lakhs", "insurance_plan",
        "genetical_risk", "normalized_risk_score"
    ]:
        df.drop("income_level", axis="columns", inplace=True)

    return df


# ---- Predict
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    use_young = _get(input_dict, "Age", "age", default=0) <= 25
    model = model_young if use_young else model_rest
    prediction = model.predict(input_df)
    return int(prediction[0])

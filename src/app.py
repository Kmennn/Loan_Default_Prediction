import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Loan Default Predictor")
st.title("Loan Default Prediction Demo")
st.write("Enter applicant details and click Predict. This demo uses a trained Random Forest model.")

# Load model
try:
    model = joblib.load("src/models/model.joblib")
except Exception as e:
    st.error("Model not found. Train the model first: python src/train_model.py")
    st.stop()

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (USD)", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount (USD)", min_value=0, value=10000)
term_months = st.selectbox("Term (months)", [36, 60])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=5)
home_ownership = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE"])
purpose = st.selectbox("Purpose", ["debt_consolidation","home_improvement","credit_card","car","small_business"])

def align_input_df(df, model):
    """
    Ensure df contains columns expected by the model's preprocessor.
    - If ColumnTransformer has explicit column lists, add missing columns with sensible defaults.
    - Returns a DataFrame with added columns in any order.
    """
    try:
        pre = model.named_steps.get("pre", None)
    except Exception:
        pre = None

    if pre is None:
        # No preprocessor detected — return as is
        return df

    # Try to read transformer column lists
    expected_cols = set()
    try:
        for name, trans, cols in pre.transformers_:
            # cols might be a list/ndarray of column names
            if isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
                for c in cols:
                    expected_cols.add(c)
    except Exception:
        # fallback: nothing to add
        expected_cols = set()

    # If expected_cols is empty, attempt another fallback: look for feature_names_in_ on pre or model
    if not expected_cols:
        if hasattr(model, "feature_names_in_"):
            expected_cols = set(model.feature_names_in_)
        elif hasattr(pre, "feature_names_in_"):
            expected_cols = set(pre.feature_names_in_)

    # Add any missing expected columns with default values
    for col in expected_cols:
        if col not in df.columns:
            # choose default based on simple heuristic
            # if name suggests id -> 0, numeric-like -> 0, else -> empty string
            if "id" in col.lower():
                df[col] = 0
            else:
                # try numeric default
                df[col] = 0

    # Reorder columns to include expected ones first (not necessary but cleaner)
    # Ensure types roughly consistent: leave to preprocessor to handle imputation/encoding
    return df

if st.button("Predict"):
    df = pd.DataFrame([{
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "purpose": purpose
    }])
    # align to model
    df = align_input_df(df, model)

    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1] if hasattr(model, "predict_proba") else None
        st.write("Prediction:", "Default" if int(pred) == 1 else "Not Default")
        if proba is not None:
            st.write("Default Probability:", f"{proba:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("If this persists, retrain the model in this environment using:")
        st.code("python src/train_model.py --input data/loan_data_sample.csv --out src/models/model.joblib")

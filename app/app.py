import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "creditbridge_model.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
    return model, feature_names, explainer

st.set_page_config(
    page_title="CreditBridge",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 CreditBridge")
st.markdown("**Explainable Credit Risk Scoring for India's Credit-Invisible Population**")
st.markdown("---")

model, feature_names, explainer = load_model()

st.sidebar.header("Applicant Details")

# Sidebar inputs
income = st.sidebar.number_input("Annual Income (₹)", 50000, 10000000, 200000, step=10000)
credit_amt = st.sidebar.number_input("Loan Amount Requested (₹)", 10000, 5000000, 500000, step=10000)
annuity = st.sidebar.number_input("Monthly Annuity (₹)", 1000, 200000, 20000, step=1000)
goods_price = st.sidebar.number_input("Goods Price (₹)", 10000, 5000000, 450000, step=10000)
age = st.sidebar.slider("Age (years)", 20, 70, 35)
employment_years = st.sidebar.slider("Years Employed", 0, 40, 5)
children = st.sidebar.slider("Number of Children", 0, 10, 0)
fam_members = st.sidebar.slider("Family Members", 1, 10, 2)
region_rating = st.sidebar.selectbox("Region Rating", [1, 2, 3], index=1)

contract_type = st.sidebar.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
gender = st.sidebar.selectbox("Gender", ["M", "F"])
own_car = st.sidebar.selectbox("Owns Car", ["Y", "N"])
own_realty = st.sidebar.selectbox("Owns Realty", ["Y", "N"])
income_type = st.sidebar.selectbox("Income Type", [
    "Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"
])
education = st.sidebar.selectbox("Education", [
    "Secondary / secondary special", "Higher education",
    "Incomplete higher", "Lower secondary", "Academic degree"
])
family_status = st.sidebar.selectbox("Family Status", [
    "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
])
housing = st.sidebar.selectbox("Housing Type", [
    "House / apartment", "Rented apartment", "With parents",
    "Municipal apartment", "Office apartment", "Co-op apartment"
])

st.sidebar.markdown("---")
st.sidebar.subheader("Credit Bureau History")
bureau_loans = st.sidebar.number_input("Total Past Loans", 0, 50, 0)
bureau_active = st.sidebar.number_input("Active Loans", 0, 20, 0)
bureau_debt = st.sidebar.number_input("Total Existing Debt (₹)", 0, 5000000, 0, step=10000)
bureau_overdue = st.sidebar.number_input("Times Overdue", 0, 30, 0)
bureau_overdue_amt = st.sidebar.number_input("Total Overdue Amount (₹)", 0, 1000000, 0, step=1000)

from sklearn.preprocessing import LabelEncoder

def encode(val, options):
    le = LabelEncoder()
    le.fit(options)
    return int(le.transform([val])[0])

is_credit_invisible = 1 if bureau_loans == 0 else 0
credit_to_income = credit_amt / (income + 1)
annuity_to_income = annuity / (income + 1)
debt_to_income = bureau_debt / (income + 1)
oldest_credit = -365 * 5 if bureau_loans > 0 else 0
recent_credit = -30 if bureau_loans > 0 else 0

input_data = {
    "AMT_INCOME_TOTAL": income,
    "AMT_CREDIT": credit_amt,
    "AMT_ANNUITY": annuity,
    "AMT_GOODS_PRICE": goods_price,
    "NAME_CONTRACT_TYPE": encode(contract_type, ["Cash loans", "Revolving loans"]),
    "CODE_GENDER": encode(gender, ["F", "M"]),
    "FLAG_OWN_CAR": encode(own_car, ["N", "Y"]),
    "FLAG_OWN_REALTY": encode(own_realty, ["N", "Y"]),
    "CNT_CHILDREN": children,
    "NAME_INCOME_TYPE": encode(income_type, [
        "Commercial associate", "Pensioner", "State servant", "Unemployed", "Working"
    ]),
    "NAME_EDUCATION_TYPE": encode(education, [
        "Academic degree", "Higher education", "Incomplete higher",
        "Lower secondary", "Secondary / secondary special"
    ]),
    "NAME_FAMILY_STATUS": encode(family_status, [
        "Civil marriage", "Married", "Separated", "Single / not married", "Widow"
    ]),
    "NAME_HOUSING_TYPE": encode(housing, [
        "Co-op apartment", "House / apartment", "Municipal apartment",
        "Office apartment", "Rented apartment", "With parents"
    ]),
    "CNT_FAM_MEMBERS": fam_members,
    "REGION_RATING_CLIENT": region_rating,
    "EXT_SOURCE_2": 0.5,
    "EXT_SOURCE_3": 0.5,
    "bureau_loan_count": bureau_loans,
    "bureau_active_loans": bureau_active,
    "bureau_total_debt": bureau_debt,
    "bureau_total_overdue": bureau_overdue_amt,
    "bureau_oldest_credit_days": oldest_credit,
    "bureau_most_recent_credit_days": recent_credit,
    "bureau_overdue_count": bureau_overdue,
    "is_credit_invisible": is_credit_invisible,
    "age_years": age,
    "employment_years": employment_years,
    "credit_to_income_ratio": credit_to_income,
    "annuity_to_income_ratio": annuity_to_income,
    "debt_to_income_ratio": debt_to_income,
}

input_df = pd.DataFrame([input_data])[feature_names]

if st.sidebar.button("Assess Credit Risk", type="primary"):
    prob = model.predict_proba(input_df)[0][1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Default Probability", f"{prob:.1%}")
    with col2:
        if prob < 0.3:
            risk = "🟢 LOW RISK"
            color = "green"
        elif prob < 0.6:
            risk = "🟡 MEDIUM RISK"
            color = "orange"
        else:
            risk = "🔴 HIGH RISK"
            color = "red"
        st.metric("Risk Tier", risk)
    with col3:
        invisible_label = "Yes — No Credit History" if is_credit_invisible else "No — Has Credit History"
        st.metric("Credit Invisible", invisible_label)

    st.markdown("---")
    st.subheader("Why this decision? — SHAP Explanation")

    shap_vals = explainer.shap_values(input_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Global Feature Importance")
    fig_path = BASE_DIR / "reports" / "figures" / "shap_summary_bar.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Top 20 Features by SHAP Importance")

else:
    st.info("👈 Fill in applicant details in the sidebar and click **Assess Credit Risk**")

    st.subheader("About CreditBridge")
    st.markdown("""
    CreditBridge addresses a critical gap in financial inclusion: **44,020 applicants (14.3%)
    in this dataset have zero formal credit history**, making them invisible to traditional
    scoring systems despite potentially being creditworthy borrowers.

    **How it works:**
    - Enter applicant details including income, loan request, and any existing credit history
    - XGBoost model predicts default probability (ROC-AUC: 0.716)
    - SHAP explainability shows exactly which factors drove the decision
    - Risk is categorized into Low / Medium / High tiers

    **Tech Stack:** PostgreSQL · XGBoost · SHAP · Streamlit · Python
    """)
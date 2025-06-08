import streamlit as st
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from src.pipelines.utils.config_list import configure

# Load paths from config
# _, _, _, _, _, _, _, _, file_path_X_train, _, _, _, _, _, _, _, model_path,shap_plot_path = configure()

model_path = "data/03_models/best_model.pkl"
file_path_X_train = "data/02_intermediate/X_train.csv"

# Load model and raw training data
model: joblib = joblib.load(model_path)
X_train = pd.read_csv(file_path_X_train)

# Exact features your SVC was trained on:
FEATURES = ['amount', 'payments', 'A4', 'A15', 'A16']

# Reconstruct the same scaler from training
scaler = StandardScaler().fit(X_train[FEATURES].astype(np.float64))

# Ensure output directory for SHAP plots exists
os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
beeswarm_path = shap_plot_path.replace('.png', '_beeswarm.png')

# ─── 2. STREAMLIT UI SETUP ─────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🏦 Loan Default Prediction with SHAP Explainability")
st.markdown(
    "Enter applicant details in the sidebar, click **Predict**, "
    "and see both the predicted default risk and a SHAP-based explanation."
)
st.sidebar.header("🔧 Input Customer Information")

def user_input() -> pd.DataFrame:
    """Collects raw inputs and returns a single-row DF in the correct order."""
    amt = st.sidebar.slider("Loan Amount (CZK)", 1_000, 1_000_000, 50_000)
    pay = st.sidebar.number_input("Monthly Payments (CZK)", 100, 50_000, 1_000)
    inh = st.sidebar.number_input("Total Inhabitants in District", 1_000, 1_000_000, 50_000)
    c95 = st.sidebar.slider("Crimes in 1995", 0, 5_000, 250)
    c96 = st.sidebar.slider("Crimes in 1996", 0, 5_000, 300)
    df = pd.DataFrame([{
        'amount': amt,
        'payments': pay,
        'A4': inh,
        'A15': c95,
        'A16': c96
    }])
    return df[FEATURES]

input_df = user_input()

# ─── 3. CACHE SHAP EXPLAINER ───────────────────────────────────────────────────
@st.cache_resource
def get_explainer():
    bg = (
        X_train[FEATURES]
        .sample(100, random_state=42)
        .astype(np.float64)
    )
    return shap.KernelExplainer(
        lambda x: model.predict_proba(
            scaler.transform(pd.DataFrame(x, columns=FEATURES))
        )[:, 1],
        bg
    )

explainer = get_explainer()

# ─── 4. PREDICT & EXPLAIN ─────────────────────────────────────────────────────
if st.button("🚀 Predict and Explain"):
    # A) Scale & predict
    scaled = scaler.transform(input_df)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.subheader("🔍 Prediction Result")
    st.write(f"**Prediction:** {'Default' if pred==1 else 'No Default'}")
    st.write(f"**Probability of Default:** {prob:.2%}")

    # B) Compute SHAP values on raw input
    shap_vals = explainer.shap_values(input_df.astype(np.float64))
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]             # class‐1 SHAP for the single row
        base = explainer.expected_value[1]
    else:
        vals = shap_vals[0]
        base = explainer.expected_value

    # C) Waterfall Plot
    st.markdown("### Individual Prediction (Waterfall Plot)")
    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=input_df.iloc[0].values,
        feature_names=FEATURES
    )
    # Draw onto current figure
    shap.plots.waterfall(exp, show=False)
    fig = plt.gcf()
    # --- SAVE WATERFALL ---
    fig.savefig(shap_plot_path, bbox_inches="tight", dpi=300)
    st.pyplot(fig)
    plt.clf()

    # D) Beeswarm Plot
    st.markdown("### Feature Importance Summary (Beeswarm Plot)")
    summary_df = (
        X_train[FEATURES]
        .sample(100, random_state=24)
        .astype(np.float64)
    )
    summary_vals = explainer.shap_values(summary_df)
    if isinstance(summary_vals, list):
        summary_vals = summary_vals[1]

    plt.figure(figsize=(8, 4))
    shap.summary_plot(summary_vals, summary_df, plot_type="dot", show=False)
    fig2 = plt.gcf()
    # --- SAVE BEESWARM ---
    fig2.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
    st.pyplot(fig2)
    plt.clf()

    st.info("🔴 High feature values push toward default — 🔵 Low values push away from default")

else:
    st.info("▶️ Fill in the sidebar and click **Predict and Explain**.")
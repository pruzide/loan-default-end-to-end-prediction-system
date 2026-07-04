# import streamlit as st
# import pandas as pd
# import os
# import joblib
# import shap
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# # from src.pipelines.utils.config_list import configure

# # Load paths from config
# # _, _, _, _, _, _, _, _, file_path_X_train, _, _, _, _, _, _, _, model_path,shap_plot_path = configure()

# model_path = "data/03_models/best_model.pkl"
# file_path_X_train = "data/02_intermediate/X_train.csv"
# shap_plot_path = "data/04_reporting/SVC_shap.png"

# # Load model and raw training data
# model: joblib = joblib.load(model_path)
# X_train = pd.read_csv(file_path_X_train)

# # Exact features your SVC was trained on:
# FEATURES = ['amount', 'payments', 'A4', 'A15', 'A16']

# # Reconstruct the same scaler from training
# scaler = StandardScaler().fit(X_train[FEATURES].astype(np.float64))

# # Ensure output directory for SHAP plots exists
# os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
# beeswarm_path = shap_plot_path.replace('.png', '_beeswarm.png')

# # ─── 2. STREAMLIT UI SETUP ─────────────────────────────────────────────────────
# st.set_page_config(layout="wide")
# st.title("🏦 Loan Default Prediction with SHAP Explainability")
# st.markdown(
#     "Enter applicant details in the sidebar, click **Predict**, "
#     "and see both the predicted default risk and a SHAP-based explanation."
# )
# st.sidebar.header("🔧 Input Customer Information")

# def user_input() -> pd.DataFrame:
#     """Collects raw inputs and returns a single-row DF in the correct order."""
#     amt = st.sidebar.slider("Loan Amount (CZK)", 1_000, 1_000_000, 50_000)
#     pay = st.sidebar.number_input("Monthly Payments (CZK)", 100, 50_000, 1_000)
#     inh = st.sidebar.number_input("Total Inhabitants in District", 1_000, 1_000_000, 50_000)
#     c95 = st.sidebar.slider("Crimes in 1995", 0, 5_000, 250)
#     c96 = st.sidebar.slider("Crimes in 1996", 0, 5_000, 300)
#     df = pd.DataFrame([{
#         'amount': amt,
#         'payments': pay,
#         'A4': inh,
#         'A15': c95,
#         'A16': c96
#     }])
#     return df[FEATURES]

# input_df = user_input()

# # ─── 3. CACHE SHAP EXPLAINER ───────────────────────────────────────────────────
# @st.cache_resource
# def get_explainer():
#     bg = (
#         X_train[FEATURES]
#         .sample(100, random_state=42)
#         .astype(np.float64)
#     )
#     return shap.KernelExplainer(
#         lambda x: model.predict_proba(
#             scaler.transform(pd.DataFrame(x, columns=FEATURES))
#         )[:, 1],
#         bg
#     )

# explainer = get_explainer()

# # ─── 4. PREDICT & EXPLAIN ─────────────────────────────────────────────────────
# if st.button("🚀 Predict and Explain"):
#     # A) Scale & predict
#     scaled = scaler.transform(input_df)
#     pred = model.predict(scaled)[0]
#     prob = model.predict_proba(scaled)[0][1]

#     st.subheader("🔍 Prediction Result")
#     st.write(f"**Prediction:** {'Default' if pred==1 else 'No Default'}")
#     st.write(f"**Probability of Default:** {prob:.2%}")

#     # B) Compute SHAP values on raw input
#     shap_vals = explainer.shap_values(input_df.astype(np.float64))
#     if isinstance(shap_vals, list):
#         vals = shap_vals[1][0]             # class‐1 SHAP for the single row
#         base = explainer.expected_value[1]
#     else:
#         vals = shap_vals[0]
#         base = explainer.expected_value

#     # C) Waterfall Plot
#     st.markdown("### Individual Prediction (Waterfall Plot)")
#     exp = shap.Explanation(
#         values=vals,
#         base_values=base,
#         data=input_df.iloc[0].values,
#         feature_names=FEATURES
#     )
#     # Draw onto current figure
#     shap.plots.waterfall(exp, show=False)
#     fig = plt.gcf()
#     # --- SAVE WATERFALL ---
#     fig.savefig(shap_plot_path, bbox_inches="tight", dpi=300)
#     st.pyplot(fig)
#     plt.clf()

#     # D) Beeswarm Plot
#     st.markdown("### Feature Importance Summary (Beeswarm Plot)")
#     summary_df = (
#         X_train[FEATURES]
#         .sample(100, random_state=24)
#         .astype(np.float64)
#     )
#     summary_vals = explainer.shap_values(summary_df)
#     if isinstance(summary_vals, list):
#         summary_vals = summary_vals[1]

#     plt.figure(figsize=(8, 4))
#     shap.summary_plot(summary_vals, summary_df, plot_type="dot", show=False)
#     fig2 = plt.gcf()
#     # --- SAVE BEESWARM ---
#     fig2.savefig(beeswarm_path, bbox_inches="tight", dpi=300)
#     st.pyplot(fig2)
#     plt.clf()

#     st.info("🔴 High feature values push toward default — 🔵 Low values push away from default")

# else:
#     st.info("▶️ Fill in the sidebar and click **Predict and Explain**.")


from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "data" / "03_models" / "best_model.pkl"
X_TRAIN_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "X_train.csv"

FEATURES = ["amount", "payments", "A4", "A15", "A16"]

FEATURE_LABELS = {
    "amount": "Loan Amount",
    "payments": "Monthly Payment",
    "A4": "District Population",
    "A15": "Crimes in 1995",
    "A16": "Crimes in 1996",
}


st.set_page_config(
    page_title="Loan Default Risk Assessment",
    page_icon="🏦",
    layout="wide",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        h1 {
            font-size: 2.15rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.2rem !important;
        }

        h2, h3 {
            font-weight: 650 !important;
        }

        .subtitle {
            color: #9ca3af;
            font-size: 1rem;
            margin-bottom: 1.4rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #262730;
            border-radius: 10px;
            padding: 14px 16px;
            background-color: rgba(255, 255, 255, 0.03);
        }

        .section-note {
            color: #9ca3af;
            font-size: 0.9rem;
        }

        .decision-box {
            border: 1px solid #262730;
            border-radius: 12px;
            padding: 16px 18px;
            background-color: rgba(255, 255, 255, 0.03);
            margin-top: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached model assets
# ---------------------------------------------------------

@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    x_train = pd.read_csv(X_TRAIN_PATH)

    scaler = StandardScaler()
    scaler.fit(x_train[FEATURES].astype(np.float64))

    return model, scaler, x_train


@st.cache_resource
def load_shap_explainer():
    model, scaler, x_train = load_model_assets()

    background = (
        x_train[FEATURES]
        .sample(min(60, len(x_train)), random_state=42)
        .astype(np.float64)
    )

    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(
            scaler.transform(pd.DataFrame(x, columns=FEATURES))
        )[:, 1],
        background,
    )

    return explainer


def predict_default(input_df: pd.DataFrame):
    model, scaler, _ = load_model_assets()

    scaled_input = scaler.transform(input_df[FEATURES].astype(np.float64))
    prediction = int(model.predict(scaled_input)[0])
    probability_default = float(model.predict_proba(scaled_input)[0][1])

    return prediction, probability_default


def get_risk_band(probability_default: float):
    if probability_default >= 0.50:
        return "High Risk", "Manual review required before approval."
    if probability_default >= 0.25:
        return "Medium Risk", "Additional checks are recommended."
    return "Low Risk", "Applicant appears low risk based on the model."


@st.cache_data(show_spinner=False)
def get_global_shap_values():
    _, _, x_train = load_model_assets()
    explainer = load_shap_explainer()

    sample_df = (
        x_train[FEATURES]
        .sample(min(120, len(x_train)), random_state=24)
        .astype(np.float64)
    )

    shap_values = explainer.shap_values(sample_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return sample_df, shap_values


def render_waterfall(input_df: pd.DataFrame):
    explainer = load_shap_explainer()

    shap_values = explainer.shap_values(input_df[FEATURES].astype(np.float64))

    if isinstance(shap_values, list):
        values = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        values = shap_values[0]
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=input_df[FEATURES].iloc[0].values,
        feature_names=FEATURES,
    )

    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)


def render_beeswarm():
    sample_df, shap_values = get_global_shap_values()

    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values,
        sample_df,
        plot_type="dot",
        show=False,
    )

    fig = plt.gcf()
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------

st.sidebar.header("Applicant Inputs")
st.sidebar.caption("Update the fields below to rescore the applicant.")

amount = st.sidebar.number_input(
    "Loan Amount",
    min_value=1_000,
    max_value=1_000_000,
    value=50_000,
    step=1_000,
)

payments = st.sidebar.number_input(
    "Monthly Payment",
    min_value=100,
    max_value=50_000,
    value=1_000,
    step=100,
)

a4 = st.sidebar.number_input(
    "District Population",
    min_value=1_000,
    max_value=2_000_000,
    value=50_000,
    step=1_000,
)

a15 = st.sidebar.number_input(
    "Crimes in 1995",
    min_value=0,
    max_value=100_000,
    value=250,
    step=50,
)

a16 = st.sidebar.number_input(
    "Crimes in 1996",
    min_value=0,
    max_value=100_000,
    value=300,
    step=50,
)

input_df = pd.DataFrame(
    [
        {
            "amount": amount,
            "payments": payments,
            "A4": a4,
            "A15": a15,
            "A16": a16,
        }
    ]
)


# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.title("Loan Default Risk Assessment")
st.markdown(
    '<div class="subtitle">Assess an applicant’s default risk and review the model drivers behind the score.</div>',
    unsafe_allow_html=True,
)

prediction, probability_default = predict_default(input_df)
risk_band, recommendation = get_risk_band(probability_default)
prediction_label = "Default" if prediction == 1 else "No Default"


# ---------------------------------------------------------
# Applicant and result section
# ---------------------------------------------------------

left_col, right_col = st.columns([1.15, 0.85])

with left_col:
    st.subheader("Applicant Details")

    display_df = input_df.rename(columns=FEATURE_LABELS)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    input_summary = pd.DataFrame(
        [
            ["Loan Amount", f"{amount:,.0f}"],
            ["Monthly Payment", f"{payments:,.0f}"],
            ["District Population", f"{a4:,.0f}"],
            ["Crimes in 1995", f"{a15:,.0f}"],
            ["Crimes in 1996", f"{a16:,.0f}"],
        ],
        columns=["Field", "Value"],
    )

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True,
    )

with right_col:
    st.subheader("Assessment Result")

    metric_col_1, metric_col_2 = st.columns(2)

    with metric_col_1:
        st.metric("Prediction", prediction_label)

    with metric_col_2:
        st.metric("Default Probability", f"{probability_default:.2%}")

    st.markdown("#### Risk Band")

    if risk_band == "High Risk":
        st.error(f"{risk_band}: {recommendation}")
    elif risk_band == "Medium Risk":
        st.warning(f"{risk_band}: {recommendation}")
    else:
        st.success(f"{risk_band}: {recommendation}")

    result_df = pd.DataFrame(
        [
            ["Prediction", prediction_label],
            ["Default Probability", f"{probability_default:.2%}"],
            ["Risk Band", risk_band],
            ["Recommendation", recommendation],
        ],
        columns=["Item", "Value"],
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "This is a model-assisted risk assessment and should not be treated as an automated credit approval decision."
    )


# ---------------------------------------------------------
# SHAP explanation section
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Model Explanation")

st.markdown(
    """
    The plots below explain how the model arrived at the current assessment.

    - **Applicant explanation:** shows which inputs moved this applicant’s score up or down.
    - **Overall feature impact:** shows how each feature behaves across a training sample.
    """
)

plot_col_1, plot_col_2 = st.columns(2)

with plot_col_1:
    st.markdown("#### Applicant Explanation")
    with st.spinner("Generating applicant explanation..."):
        render_waterfall(input_df)

with plot_col_2:
    st.markdown("#### Overall Feature Impact")
    with st.spinner("Generating beeswarm plot..."):
        render_beeswarm()

st.caption(
    "SHAP values show the contribution of each feature toward or away from predicted default risk."
)
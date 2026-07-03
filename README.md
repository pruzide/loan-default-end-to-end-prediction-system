# 🏦 Loan Default Prediction System with SHAP Explainability

This end-to-end machine learning project predicts whether a customer will default on a loan using the Czech financial dataset. It includes model training, explainability with SHAP, and a deployed UI on Streamlit.

I suggest you to go through the Jupyter Notebook to go through the detailed process of selection of model and experimenting with various features and finalizing SVC as the best model. The notebook also has detailed EDA to capture various potential trends which increase credit risk.

---

## Live Demo

Primary deployment:

- AWS EC2: http://13.53.125.154:8501

Fallback/staging:

- Streamlit Community Cloud: https://loan-default-end-to-end-prediction-system-g6wlk74appi6gwvxjzee.streamlit.app/

---


## Metrics

The deployed baseline SVC model is evaluated using a stratified 70/30 train-test split with `random_state=42`.

| Metric | Value |
|---|---:|
| Accuracy | 69.76% |
| Default-class precision | 21.74% |
| Default-class recall | 65.22% |

Full report:

- [`reports/model_metrics.json`](reports/model_metrics.json)



## Docker

Build the Streamlit image:

```bash
docker build -t loan-default-app .

Additional leakage-safe tuning experiments using pre-loan transaction features are available in the reports folder. These experiments showed that the original resume metric range is not supported by the current dataset/model setup without changing the modelling definition.


```markdown
## Inference API

Health check:

```bash
curl http://localhost:8000/health


```markdown
## AWS EC2 Deployment

The Dockerized Streamlit dashboard and FastAPI inference service can be deployed on Ubuntu EC2 using:

- [`deploy/AWS_EC2.md`](deploy/AWS_EC2.md)


## Load Testing

Load testing is performed against the FastAPI inference endpoint using concurrent requests.

Report:

- [`reports/load_test_results.md`](reports/load_test_results.md)

The EC2 load test validates single-customer scoring at 15–25 concurrent users. The latency claim should be read from the latest p95 result in the report.



## ⚙️ Features

- 📊 **Data Cleaning** and feature engineering from relational financial tables
- 🧠 **SVC Classifier** with scaling and class balancing
- 📈 **SHAP Explainability** (KernelExplainer with Waterfall + Beeswarm plots)
- 🧪 **Evaluation Reports** with precision/recall/F1-score
- 🌐 **Streamlit-based UI** + SHAP visualization support
- 🔁 **Fully modular pipeline** with reusability

---

## 📦 Technologies Used

- Python (3.10)
- scikit-learn
- SHAP
- Pandas, NumPy, Matplotlib
- Streamlit
- Joblib

---

## 🛠 Setup

Clone and install

git clone https://github.com/pruzide/loan-default-end-to-end-prediction-system.git

cd loan-default-end-to-end-prediction-system

pip install -r requirements.txt

---


# Run app
streamlit run app.py

---


## 🧠 SHAP Outputs
📄 Waterfall plots for individual prediction explanation

📊 Beeswarm plots for global feature impact

ℹ️ Background dataset used from training set (**scaled + unscaled input**)

---

## 📄 License
This project is under the MIT License.






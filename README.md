# 🏦 Loan Default Prediction System with SHAP Explainability

This end-to-end machine learning project predicts whether a customer will default on a loan using the Czech financial dataset. It includes model training, explainability with SHAP, and a deployed UI on Streamlit.

---

## 🚀 Live Demo
Access the deployed app here:  
👉 [Streamlit App](https://loan-default-end-to-end-prediction-system-g6wlk74appi6gwvxjzee.streamlit.app/)

---

## 📂 Project Structure

loan-default-end-to-end-prediction-system/
├── app.py # Streamlit/Gradio frontend app
├── Dockerfile # For deployment via Hugging Face
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── data/
│ ├── 01_raw/ # Raw data files
│ ├── 02_intermediate/ # Processed training/testing sets
│ └── 03_models/ # Trained models (.pkl)
├── notebooks/ # Jupyter notebooks for EDA & training
├── src/
│ └── pipelines/
│ ├── data_preprocessing.py # Data cleaning & transformation logic
│ ├── modelling.py # Model training and SHAP logic
│ └── utils/
│ └── config_list.py # Centralized path and feature config



---

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
- Streamlit / Gradio
- Joblib

---

## 🛠 Setup

```bash
# Clone and install
git clone https://github.com/pruzide/loan-default-end-to-end-prediction-system.git
cd loan-default-end-to-end-prediction-system
pip install -r requirements.txt

# Run app
streamlit run app.py


🧠 SHAP Outputs
📉 Waterfall plots for individual prediction

🐝 Beeswarm plots for global feature impact

Background dataset used from training set (scaled + unscaled input)


📄 License
This project is under the MIT License.






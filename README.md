# 💸 Loan Default End-to-End Prediction System

An **end-to-end machine learning system** to predict the risk of **loan default** using structured banking data from the Czech Republic (Berka dataset). This project combines **model training**, **SHAP-based explainability**, and **interactive frontend deployment** using **Streamlit** and **Gradio**.

---

## 🚀 Live Demo  
🔗 [Streamlit App](https://loan-default-end-to-end-prediction-system-g6wlk74appi6gwvxjzee.streamlit.app/)

---

## 🧠 Problem Statement  
Banks need to assess creditworthiness of applicants efficiently. This system predicts **whether a customer is likely to default on a loan** using features such as loan amount, district data, and criminal records.

---

## 🛠️ Tech Stack

| Layer | Tools Used |
|-------|------------|
| **Data Processing** | `pandas`, `numpy` |
| **Modeling** | `scikit-learn` (SVC with `StandardScaler`) |
| **Explainability** | `SHAP` (KernelExplainer + plots) |
| **Frontend** |  `Streamlit` |
| **Deployment** | `Streamlit Cloud` |
| **Monitoring** | UptimeRobot / BetterStack |

---

## 🔍 Features

- ✅ Cleaned & engineered features from raw bank datasets  
- ✅ Trained **SVC classifier** on scaled features  
- ✅ SHAP explainability (Waterfall + Beeswarm plots)  
- ✅ Public **interactive UI** to input customer details  
- ✅ **Deployed and monitored** in production  

---

## 📊 Key Input Features

| Feature | Description |
|---------|-------------|
| `amount` | Total loan amount requested |
| `payments` | Monthly installment amount |
| `A4` | Total inhabitants in applicant's district |
| `A15` | Number of crimes in district (1995) |
| `A16` | Number of crimes in district (1996) |

---

## 📈 Output

- **Prediction**: `Default` or `No Default`  
- **Probability of Default**: e.g., 27.4%  
- **SHAP Explanations**: Visual insights into which features influenced the prediction

---

## 🗂️ Project Structure
.
├── app.py # Streamlit frontend
├── data/
│ ├── 02_intermediate/ # Processed training data
│ └── 03_models/ # Trained SVC model
├── src/
│ └── pipelines/utils/ # config_list.py and helpers
├── requirements.txt
└── README.md


---

## 📌 How to Run Locally

```bash
git clone https://github.com/pruzide/loan-default-end-to-end-prediction-system.git
cd loan-default-end-to-end-prediction-system
pip install -r requirements.txt
streamlit run app.py

👨‍💻 Author
Gourav Singh
Github | LinkedIn




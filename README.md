# ⚡ VoltIQ - Smart AI Electricity & Solar Audit Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://voltiq.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**VoltIQ** is an end-to-end Machine Learning web platform that predicts daily household/building electricity consumption (in kWh), estimates monthly electricity bills (₹), evaluates carbon footprint impact, recommends optimal solar panel system sizing, and calculates investment return (ROI & Payback Period).

🌐 **Live Web Application**: [https://voltiq.streamlit.app](https://voltiq.streamlit.app)

---

## 🌟 Key Features

* ⚡ **Daily Consumption Prediction (kWh)**: Powered by a calibrated **Random Forest Regressor** model.
* 💵 **Bill Calculator**: Estimates monthly and annual electricity expenses based on customizable per-unit tariff rates (₹ / kWh).
* ☀️ **Solar System Sizing & Payback Roadmap**: Recommends required kW solar capacity and computes net investment after PM Surya Ghar Govt Subsidies.
* 🍃 **Carbon Footprint Audit**: Calculates monthly/annual grid CO₂ emissions and equivalent trees saved per year.
* ☀️ ⛅ ❄️ **Seasonal Bill Comparison**: Visualizes bill variations across Summer, Monsoon, and Winter using interactive **Plotly** bar charts.
* 📄 **Multilingual PDF Report Export**: Generates downloadable Electricity & Solar Audit Statements in **English**, **मराठी (Marathi)**, and **हिंदी (Hindi)**.
* 🎨 **Custom Branding**: Official audit report generated with `Powered by Rahul Mahanavar Software` signoff.

---

## 🏗️ Project Architecture & Pipeline

Following an end-to-end Machine Learning Workflow:

1. **Step 1: Understanding the Dataset**: Features include Temperature, Humidity, Building Area, Occupants, Appliance Count, Weekend Flag, and AC Usage Hours.
2. **Step 2: Library Selection**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `reportlab`, `plotly`.
3. **Step 3: Data Loading**: Calibrated Indian household electricity consumption dataset (`electricity_consumption.csv`).
4. **Step 4: Exploration & Cleaning**: Statistical analysis, null value validation (`info()`, `describe()`).
5. **Step 5: Exploratory Data Analysis (EDA)**: Histogram distribution plots and Pearson correlation heatmaps.
6. **Step 6: Data Preprocessing**: Train-test split (80-20) and standardization via `StandardScaler`.
7. **Step 7: Model Training & Evaluation**: Comparing Linear Regression, Ridge, Lasso, Decision Tree, and Random Forest models (evaluated using MSE, MAE, R² Score).
8. **Step 8: Model Persistence**: Exporting fitted `best_model.pkl` and `scaler.pkl` using `pickle`.
9. **Step 9: Streamlit Cloud Deployment**: Interactive web application (`app.py`).

---

## 💻 Tech Stack

- **Frontend & Web Framework**: Streamlit
- **Machine Learning**: Scikit-Learn (Random Forest Regressor)
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Data Visualization**: Plotly Express, Seaborn, Matplotlib
- **Document Generation**: ReportLab (Multilingual Devanagari PDF Engine)

---

## 🚀 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rahulmaha125/VoltIQ.git
   cd VoltIQ

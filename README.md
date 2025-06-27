# 📊 COGS Prediction App

A machine learning-powered web application built with **Streamlit** that predicts **Cost of Goods Sold (COGS)** based on business and product sales data.

This app uses a trained **Random Forest Regressor** model to forecast COGS from inputs like Units Sold, Sales Price, Gross Sales, Discounts, and more.

---

## 🚀 Live Demo

👉 [Click to Launch the App](https://your-username-cogs-prediction-app.streamlit.app)

---

## 📸 App Preview

![COGS App Screenshot](images/overview.png)

---

## 🎯 Features

- 📥 Input product and sales data through an interactive form
- 🤖 Model predicts Cost of Goods Sold (COGS) instantly
- 📊 Built with real-world business analytics use-case
- 🌐 Deployed on Streamlit Cloud for live access

---

## 🧠 Machine Learning Model

- Model: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- Trained on business sales data
- Target Variable: `COGS`
- Input Features:  
  - Units Sold  
  - Sale Price  
  - Gross Sales  
  - Discounts  
  - Sales  
  - Profit  
  - Segment, Country, Product, Discount Band, Quarter, Year, Month  

---

## 🛠️ Tech Stack

- **Python** 🐍
- **Streamlit** 🌐
- **Scikit-learn**
- **Pandas**
- **Joblib**
- **Git & GitHub**

---

## 📁 Project Structure

Cost-Prediction-Business-Budgeting/

│

├── app.py # Streamlit web app

├── models/

│ ├── random_forest_cogs.pkl

│ └── feature_columns.pkl

├── requirements.txt # Dependencies

├── README.md # Project documentation

└── images/

└── overview.png # App screenshot or diagram


---

## ⚙️ Installation & Running Locally

1. **Clone this repo**:
   ```bash
   git clone https://github.com/yourusername/cogs-prediction-app.git
   cd cogs-prediction-app

   This app is deployed on Streamlit Cloud and accessible from any device with internet access.

# 👤 Author

## Praise Dunmininu

Machine Learning Enthusiast | Data Scientist | Tech for Impact in Africa

📧 [Your Email]

🌐 [LinkedIn Profile]

🛠️ Built as part of a client-facing business analysis support tool

# 🤝 License

This project is open source and free to use under the MIT License.



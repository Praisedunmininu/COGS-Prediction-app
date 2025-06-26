import streamlit as st
import pandas as pd
import joblib

# Load model + columns
model = joblib.load("models/random_forest_cogs.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

st.set_page_config(page_title="COGS Predictor", layout="centered")

st.title("📊 Predict Cost of Goods Sold (COGS)")
st.markdown("Fill in the fields below to predict your COGS using a trained Random Forest model.")

# Categorical inputs
segment = st.selectbox("Segment", ["Government", "Midmarket", "Enterprise", "Channel Partners", "Small Business"])
country = st.selectbox("Country", ["Canada", "Germany", "France", "Mexico", "United States"])
product = st.selectbox("Product", ["Carretera", "Montana", "Paseo", "Velo"])
discount_band = st.selectbox("Discount Band", ["None", "Low", "Medium", "High"])
quarter = st.selectbox("Quarter", ["2014Q1", "2014Q2", "2014Q3", "2014Q4"])

# Numeric inputs
units_sold = st.number_input("Units Sold", min_value=0.0)
mfg_price = st.number_input("Manufacturing Price ($)", min_value=0.0)
sale_price = st.number_input("Sale Price ($)", min_value=0.0)
gross_sales = st.number_input("Gross Sales ($)", min_value=0.0)
discounts = st.number_input("Discounts ($)", min_value=0.0)
sales = st.number_input("Sales ($)", min_value=0.0)
profit = st.number_input("Profit ($)", min_value=0.0)
year = st.selectbox("Year", [2014])
month = st.selectbox("Month", list(range(1, 13)))

# Button to predict
if st.button("🔍 Predict COGS"):

    # Make input row
    input_dict = {
        'Units Sold': units_sold,
        'Manufacturing Price': mfg_price,
        'Sale Price': sale_price,
        'Gross Sales': gross_sales,
        'Discounts': discounts,
        'Sales': sales,
        'Profit': profit,
        'year': year,
        'month': month,
        'Segment_' + segment: 1,
        'Country_' + country: 1,
        'Product_' + product: 1,
        'Discount Band_' + discount_band: 1,
        'quarter_' + quarter: 1,
    }

    # Create full row with 0s
    input_data = pd.DataFrame([[0]*len(feature_cols)], columns=feature_cols)
    for key, val in input_dict.items():
        if key in input_data.columns:
            input_data[key] = val

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"🧾 Predicted COGS: **${prediction:,.2f}**")

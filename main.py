import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and preprocessing tools
rf_regressor = joblib.load('regression_model_rf.joblib')
rf_classifier = joblib.load('classification_model_rf.joblib')
kmeans_model = joblib.load('clustering_model_kmeans.joblib')
encoder = joblib.load('target_encoder.joblib')
scaler = joblib.load('scaler.joblib')

st.set_page_config(page_title="Import Data ML App", page_icon="📊")

st.title("📦 Import Data ML Mini Project")
st.markdown("Predict, classify, and cluster import data using trained ML models.")

# Sidebar menu
menu = st.sidebar.radio("Select Task", ["Regression", "Classification", "Clustering"])

# Common input fields
st.sidebar.header("Input Data")
value_qt = st.sidebar.number_input("Value Quantity (value_qt)", value=1000.0)
year = st.sidebar.number_input("Year", value=2023, step=1)
month = st.sidebar.slider("Month", 1, 12, 6)
country_top = st.sidebar.text_input("Country Name (e.g., USA)", "USA")
commodity_top = st.sidebar.text_input("Commodity Name", "Oil")

input_df = pd.DataFrame({
    'value_qt': [value_qt],
    'year': [year],
    'month': [month],
    'country_top': [country_top],
    'commodity_top': [commodity_top]
})

# Encode and scale
enc_df = encoder.transform(input_df.copy())
num_cols = ['value_qt', 'year', 'month']
enc_df[num_cols] = scaler.transform(enc_df[num_cols])

# ----- REGRESSION -----
if menu == "Regression":
    st.subheader("💰 Predict Import Value (Regression)")
    pred_val = rf_regressor.predict(enc_df)[0]
    st.success(f"Predicted Import Value (₹): {pred_val:,.2f}")

# ----- CLASSIFICATION -----
elif menu == "Classification":
    st.subheader("📈 Classify Import Value (Low / Medium / High)")
    pred_class = rf_classifier.predict(enc_df)[0]
    st.success(f"Predicted Category: {pred_class.upper()}")

# ----- CLUSTERING -----
elif menu == "Clustering":
    st.subheader("🌍 Cluster Country by Trade Pattern")
    # Simple mock clustering logic
    X = enc_df[['value_qt','year','month']].copy()
    cluster_label = kmeans_model.predict(scaler.transform(X))[0]
    st.info(f"Country '{country_top}' belongs to cluster {cluster_label}")

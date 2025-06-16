import streamlit as st
import pandas as pd

st.set_page_config(page_title="E-Commerce Anomaly Detection", layout="wide")

st.title("E-Commerce Anomaly Detection Dashboard")

df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])

selected_model = st.selectbox("Select Anomaly Detection Model", ["Isolation Forest", "One-Class SVM"])

anomaly_column = "anomaly_if" if selected_model == "Isolation Forest" else "anomaly_svm"

st.sidebar.header("Filters")
min_amount = st.sidebar.slider("Minimum Amount", float(df["amount"].min()), float(df["amount"].max()), float(df["amount"].min()))
max_amount = st.sidebar.slider("Maximum Amount", float(df["amount"].min()), float(df["amount"].max()), float(df["amount"].max()))

filtered_df = df[(df["amount"] >= min_amount) & (df["amount"] <= max_amount)]

st.dataframe(filtered_df[filtered_df[anomaly_column] == -1], use_container_width=True)

st.markdown("Rows shown are anomalous transactions based on the selected model.")
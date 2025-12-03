import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="SupplySense 360", layout="wide")

# Load processed data
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/supply_data.csv")

df = load_data()

# Header
st.title("SupplySense 360")
st.subheader("Supply Chain Performance Dashboard")

# KPI Cards
col1, col2, col3 = st.columns(3)

# Select some numeric columns to show as KPI cards
numeric_cols = df.select_dtypes(include='number').columns

col1.metric("Total Rows", f"{len(df):,}")
col2.metric("Numeric Columns", f"{len(numeric_cols)}")
col3.metric("Total Numeric Values", f"{df[numeric_cols].sum().sum():,.0f}")

# Data preview table
st.markdown("### Data Preview")
st.dataframe(df.head())

# Simple chart for exercise
if len(numeric_cols) >= 1:
    st.markdown("### Example Chart")
    fig = px.histogram(df, x=numeric_cols[0])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric columns available for charting.")




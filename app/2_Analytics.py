import streamlit as st
import pandas as pd
import plotly.express as px

st.title(" Analytics")

df = pd.read_csv("data/processed/supply_data.csv")

numeric_cols = df.select_dtypes(include='number').columns

if len(numeric_cols) > 0:
    fig = px.histogram(df, x=numeric_cols[0])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric data available.")

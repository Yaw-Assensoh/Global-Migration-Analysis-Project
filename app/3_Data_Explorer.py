import streamlit as st
import pandas as pd

st.title("🧹 Data Explorer")

df = pd.read_csv("data/processed/supply_data.csv")

st.dataframe(df)

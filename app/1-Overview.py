import streamlit as st
import pandas as pd

st.title(" Supply Overview")

df = pd.read_csv("data/processed/supply_data.csv")

st.subheader("Dataset Summary")
st.write(df.describe())

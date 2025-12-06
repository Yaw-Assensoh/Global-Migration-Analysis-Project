# File: app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Migration Analysis", layout="wide")

st.title("🌍 Global Migration Analysis")
st.markdown("**Working App - Deployment Successful**")

# Generate sample data
@st.cache_data
def get_data():
    dates = pd.date_range('2000-01-01', '2023-01-01', freq='Y')
    data = pd.DataFrame({
        'date': dates,
        'migration': np.random.randn(len(dates)).cumsum() + 50
    })
    return data

data = get_data()

# Show data
st.subheader("Sample Migration Data")
st.dataframe(data)

# Simple chart
st.subheader("Trend Chart")
st.line_chart(data.set_index('date'))

# Show package versions
st.subheader("System Status")
st.write(f"✅ Streamlit: {st.__version__}")
st.write(f"✅ Pandas: {pd.__version__}")
st.write(f"✅ NumPy: {np.__version__}")

st.balloons()
st.success("DEPLOYMENT WORKING!")
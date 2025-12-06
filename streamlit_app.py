# File: streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Migration Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Global Migration Dashboard")
st.markdown("**If you see this, deployment is WORKING!**")

# Simple test of imports
try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    
    st.success("✅ All imports successful!")
    
    # Show versions
    st.write("**Package versions:**")
    st.code(f"""
    Streamlit: {st.__version__}
    Pandas: {pd.__version__}
    NumPy: {np.__version__}
    Plotly: {go.__version__}
    """)
    
    # Create simple chart
    st.subheader("Sample Migration Chart")
    data = pd.DataFrame({
        'Year': range(2000, 2024),
        'Migration': np.random.randn(24).cumsum() + 100
    })
    st.line_chart(data.set_index('Year'))
    
except Exception as e:
    st.error(f"❌ Import failed: {e}")

st.balloons()
st.success("🎉 DEPLOYMENT SUCCESSFUL!")

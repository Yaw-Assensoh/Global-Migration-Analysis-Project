import streamlit as st

st.set_page_config(
    page_title="Global Migration Analysis",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Global Migration Analysis Dashboard")
st.markdown("""
### Advanced Migration Forecasting & Analysis

Navigate to different pages using the sidebar:

**📈 Advanced Forecasting** - Multi-model forecasting with Prophet, ARIMA, and ensemble methods
**🌍 Migration Dashboard** - Interactive visualizations and maps
**📊 Data Analysis** - Statistical insights and trend analysis

---

### Deployment Status:
✅ Config files ready  
✅ Requirements optimized  
✅ Ready for Streamlit Cloud deployment

---
**Note:** The first deployment might take a few minutes as it installs Prophet and other ML packages.
""")

# Add a success message
st.success("App configured successfully! Deploy to Streamlit Cloud when ready.")

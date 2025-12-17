# pages/06_Reports.py
import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Reports", layout="wide")

st.title(" Analysis Reports")
st.markdown("Generate professional reports and summaries")

# Report templates
st.subheader("Report Templates")

template_col1, template_col2, template_col3 = st.columns(3)

with template_col1:
    st.markdown("###  Executive Summary")
    st.markdown("""
    High-level overview for stakeholders
    - Key metrics
    - Top findings
    - Recommendations
    """)
    if st.button("Generate Executive Summary", key="exec_summary", use_container_width=True):
        st.success("Executive summary generated!")
        with st.expander("Preview Report"):
            st.markdown("""
            # Executive Summary
            **Date:** """ + datetime.now().strftime("%Y-%m-%d") + """
            
            ## Key Findings
            1. United States leads in immigration (1.2M)
            2. India leads in emigration (495K)
            3. Clear developing→developed country flow
            
            ## Recommendations
            - Focus on integration policies
            - Monitor emerging corridors
            - Prepare for demographic shifts
            """)

with template_col2:
    st.markdown("### Trends Report")
    st.markdown("""
    Detailed trend analysis
    - Historical patterns
    - Future projections
    - Risk assessment
    """)
    if st.button("Generate Trends Report", key="trends_report", use_container_width=True):
        st.success("Trends report generated!")

with template_col3:
    st.markdown("###  Regional Analysis")
    st.markdown("""
    Regional migration patterns
    - Continental trends
    - Country comparisons
    - Policy implications
    """)
    if st.button("Generate Regional Report", key="regional_report", use_container_width=True):
        st.success("Regional report generated!")

# Custom Report Generator
st.subheader("Custom Report Generator")

with st.form("custom_report"):
    report_title = st.text_input("Report Title", "Migration Analysis Report")
    
    col1, col2 = st.columns(2)
    with col1:
        include_metrics = st.checkbox("Include Key Metrics", True)
        include_trends = st.checkbox("Include Trends", True)
    with col2:
        include_forecasts = st.checkbox("Include Forecasts", False)
        include_recommendations = st.checkbox("Include Recommendations", True)
    
    submitted = st.form_submit_button("Generate Custom Report")
    
    if submitted:
        st.success("Custom report generated!")
        
        # Display report preview
        with st.expander("Report Preview", expanded=True):
            st.markdown(f"# {report_title}")
            st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            st.markdown("---")
            
            if include_metrics:
                st.markdown("## Key Metrics")
                st.markdown("""
                - Total Countries Analyzed: 233
                - Global Population: 8.1B
                - Net Migration Balance: +750K
                - Top Immigration Country: United States (1.2M)
                - Top Emigration Country: India (495K)
                """)
            
            if include_trends:
                st.markdown("## Trends Analysis")
                st.markdown("""
                - Urbanization correlates with immigration (r=0.65)
                - Aging populations require immigration for stability
                - Economic development drives emigration reduction
                """)
            
            if include_forecasts:
                st.markdown("## Forecasts")
                st.markdown("""
                - Projected immigration increase: 15% by 2030
                - Urban migration expected to grow
                - Regional disparities likely to persist
                """)
            
            if include_recommendations:
                st.markdown("## Recommendations")
                st.markdown("""
                1. **For Destination Countries:**
                   - Develop integration programs
                   - Plan urban infrastructure
                   - Monitor demographic balance
                
                2. **For Source Countries:**
                   - Create economic opportunities
                   - Invest in education
                   - Facilitate circular migration
                
                3. **For International Organizations:**
                   - Foster cooperation agreements
                   - Share best practices
                   - Monitor migration corridors
                """)
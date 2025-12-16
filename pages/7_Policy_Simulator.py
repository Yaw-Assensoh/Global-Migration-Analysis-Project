# pages/7_Policy_Simulator.py - DASHBOARD VERSION
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================== PAGE SETUP ====================
st.set_page_config(
    page_title="Policy Simulator",
    page_icon="",
    layout="wide"
)

# Use same styling as your existing dashboard
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E86AB;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("Policy Impact Simulator")
st.markdown("Test how policy changes affect migration forecasts")
st.markdown('</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Country selection - matches your forecasting dashboard
    countries = ["USA", "Germany", "India", "Nigeria", "Japan", "Brazil"]
    selected_country = st.selectbox("🌍 Select Country", countries)
    
    st.markdown("---")
    st.markdown("###  Policy Changes")
    
    # Simple policy sliders
    visa_change = st.slider(
        "Visa Policy Change",
        -30, 30, 0, 5,
        help="Make visas easier (+) or harder (-)"
    )
    
    work_change = st.slider(
        "Work Permit Change", 
        -30, 30, 0, 5,
        help="Make work permits easier (+) or harder (-)"
    )
    
    st.markdown("---")
    
    # Run button
    run_simulation = st.button(" Run Simulation", type="primary", use_container_width=True)

# ==================== MAIN DASHBOARD ====================
if run_simulation:
    # Create two columns for dashboard layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main visualization
        st.markdown(f"### Policy Impact: {selected_country}")
        
        # Generate sample data (in real app, use your actual forecast data)
        historical_years = list(range(2020, 2025))
        forecast_years = list(range(2025, 2030))
        
        # Base migration rate by country
        base_rates = {
            "USA": 3.1, "Germany": 1.5, "India": -0.4,
            "Nigeria": -0.2, "Japan": 0.3, "Brazil": -0.2
        }
        
        base_rate = base_rates.get(selected_country, 2.0)
        
        # Generate historical data
        historical_rates = []
        for i in range(5):
            rate = base_rate + (i * 0.05) + np.random.normal(0, 0.1)
            historical_rates.append(round(rate, 2))
        
        # Generate baseline forecast (from your forecasting model)
        baseline_forecast = []
        last_rate = historical_rates[-1]
        for i in range(5):
            rate = last_rate * (1.01 ** (i + 1))
            baseline_forecast.append(round(rate, 2))
        
        # Apply policy impact (simple formula)
        total_change = (visa_change * 0.6 + work_change * 0.4) / 100
        policy_impact = 1 + total_change
        
        # Generate scenario forecast
        scenario_forecast = []
        for i, base_rate in enumerate(baseline_forecast):
            # Impact diminishes over time
            diminishing = 0.9 ** i
            adjusted_impact = 1 + ((policy_impact - 1) * diminishing)
            scenario_forecast.append(round(base_rate * adjusted_impact, 2))
        
        # Create the chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_years,
            y=historical_rates,
            mode='lines+markers',
            name='Historical',
            line=dict(color='#2E86AB', width=3)
        ))
        
        # Baseline forecast
        fig.add_trace(go.Scatter(
            x=forecast_years,
            y=baseline_forecast,
            mode='lines',
            name='Baseline Forecast',
            line=dict(color='gray', width=3, dash='dash')
        ))
        
        # Scenario forecast
        fig.add_trace(go.Scatter(
            x=forecast_years,
            y=scenario_forecast,
            mode='lines',
            name='With Policy Changes',
            line=dict(color='#FF6B6B', width=4)
        ))
        
        # Add impact area
        fig.add_trace(go.Scatter(
            x=forecast_years + forecast_years[::-1],
            y=scenario_forecast + baseline_forecast[::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Impact Area',
            showlegend=False
        ))
        
        # Layout
        fig.update_layout(
            title=f'Impact of Policy Changes on Migration',
            xaxis_title='Year',
            yaxis_title='Migration Rate',
            height=500,
            plot_bgcolor='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Impact metrics
        st.markdown("### Impact Summary")
        
        # Calculate impacts
        final_impact = scenario_forecast[-1] - baseline_forecast[-1]
        cumulative_impact = sum(s - b for s, b in zip(scenario_forecast, baseline_forecast))
        percent_change = (final_impact / baseline_forecast[-1]) * 100
        
        # Display metrics
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Final Year Impact</div>
            <div class="metric-value">{final_impact:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">5-Year Total</div>
            <div class="metric-value">{cumulative_impact:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Percentage Change</div>
            <div class="metric-value">{percent_change:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Policy summary
        st.markdown("### Policy Changes")
        st.write(f"Visa Policy: {visa_change:+} points")
        st.write(f"Work Permits: {work_change:+} points")
        
        # Quick recommendation
        if final_impact > 0.2:
            st.success("✅ Significant positive impact")
        elif final_impact > 0:
            st.info(" Moderate positive impact")
        elif final_impact < -0.2:
            st.error("⚠️ Significant negative impact")
        else:
            st.warning("➡️ Minimal impact")
    
    # Bottom section - forecast table
    st.markdown("---")
    st.markdown("### Forecast Comparison")
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Year': forecast_years,
        'Baseline': baseline_forecast,
        'With Policy Changes': scenario_forecast,
        'Difference': [round(s - b, 2) for s, b in zip(scenario_forecast, baseline_forecast)]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

else:
    # Landing page
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Policy Impact Simulator
        
        This tool lets you test how changes in migration policies affect future migration patterns.
        
        **How it works:**
        1. Select a country from the sidebar
        2. Adjust the policy sliders
        3. Click "Run Simulation"
        4. See the impact on migration forecasts
        
        **What you can test:**
        - Make visas easier or harder
        - Make work permits easier or harder
        - See the combined effect on migration
        
        **The dashboard shows:**
        - Chart comparing baseline vs. policy scenario
        - Impact metrics (final year, 5-year total, % change)
        - Forecast comparison table
        """)
    
    with col2:
        st.markdown("""
        ### Quick Example
        
        Try testing:
        
        **Easier Migration Policies:**
        - Visa: +15
        - Work Permits: +10
        
        **Stricter Migration Policies:**
        - Visa: -15
        - Work Permits: -10
        
        ---
        
        **Ready to test?**
        
        Configure settings in the sidebar and click "Run Simulation"
        """)

# Footer
st.markdown("---")
st.caption("Policy Simulator • Part of Migration Forecasting Dashboard")
# ==================== CELL 11: STREAMLIT DASHBOARD INTEGRATION CODE ====================
# Generate code for integrating these forecasts into your Streamlit dashboard


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# Load pre-trained forecasting models
@st.cache_resource
def load_forecast_models():
    """Load pre-trained forecasting models"""
    models = {}
    try:
        # Load your trained models here
        # Example: models['arima'] = pickle.load(open('models/arima_model.pkl', 'rb'))
        # Example: models['prophet'] = pickle.load(open('models/prophet_model.pkl', 'rb'))
        pass
    except Exception as e:
        st.warning(f"Could not load models: {e}")
    return models

def generate_forecast(country, forecast_horizon, confidence_level):
    """Generate forecast for selected country"""
    # This would call your trained models
    # Return forecast values and confidence intervals
    
    # Mock implementation
    forecast_years = list(range(2026, 2026 + forecast_horizon))
    base_value = 5.0  # This would come from your data
    
    # Generate forecast with some trend
    forecast = [base_value * (1 + 0.02*i) for i in range(forecast_horizon)]
    
    # Generate confidence intervals based on confidence level
    if confidence_level == 0.68:
        z_score = 1.0
    elif confidence_level == 0.80:
        z_score = 1.28
    else:  # 0.95
        z_score = 1.96
    
    ci_width = [0.5 * (1 + 0.1*i) for i in range(forecast_horizon)]  # Increasing uncertainty
    lower = [f - z_score * w for f, w in zip(forecast, ci_width)]
    upper = [f + z_score * w for f, w in zip(forecast, ci_width)]
    
    return forecast_years, forecast, lower, upper

def create_forecast_visualization(historical_years, historical_values,
                                  forecast_years, forecast_values,
                                  lower_bounds, upper_bounds,
                                  country_name, confidence_level):
    """Create interactive forecast visualization"""
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_years,
        y=historical_values,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_years,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_years + forecast_years[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        name=f'{int(confidence_level*100)}% Confidence Interval'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Migration Forecast for {country_name}',
        xaxis_title='Year',
        yaxis_title='Migration Rate (per 1000)',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='white'
    )
    
    # Add vertical line separating historical and forecast
    fig.add_vline(x=historical_years[-1] + 0.5, line_width=2, line_dash="dash", line_color="gray")
    
    return fig

def main():
    st.title("Advanced Migration Forecasting")
    
    # Sidebar controls
    st.sidebar.header("Forecast Settings")
    
    # Country selection
    available_countries = ["United States", "Germany", "India", "Nigeria", "Japan", "Brazil"]
    selected_country = st.sidebar.selectbox("Select Country", available_countries)
    
    # Forecast horizon
    forecast_horizon = st.sidebar.slider("Forecast Horizon (years)", 1, 10, 5)
    
    # Confidence level
    confidence_level = st.sidebar.select_slider(
        "Confidence Level",
        options=[0.68, 0.80, 0.95],
        value=0.95,
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    # Model selection
    model_type = st.sidebar.radio(
        "Forecast Model",
        ["Ensemble (Recommended)", "ARIMA", "Prophet", "Linear"]
    )
    
    # Generate forecast
    if st.sidebar.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Load historical data for selected country
            # historical_data = load_country_data(selected_country)
            
            # Mock historical data
            historical_years = list(range(2015, 2026))
            historical_values = [5 + np.random.normal(0, 0.5) for _ in historical_years]
            
            # Generate forecast
            forecast_years, forecast_values, lower_bounds, upper_bounds = generate_forecast(
                selected_country, forecast_horizon, confidence_level
            )
            
            # Create visualization
            fig = create_forecast_visualization(
                historical_years, historical_values,
                forecast_years, forecast_values,
                lower_bounds, upper_bounds,
                selected_country, confidence_level
            )
            
            # Display forecast
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast table
            st.subheader("Forecast Values")
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                'Forecast': forecast_values,
                f'Lower Bound ({int(confidence_level*100)}%)': lower_bounds,
                f'Upper Bound ({int(confidence_level*100)}%)': upper_bounds
            })
            st.dataframe(forecast_df.style.format("{:.2f}"))
            
            # Display forecast summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("5-Year Average Forecast", f"{np.mean(forecast_values):.2f}")
            with col2:
                st.metric("Uncertainty Range", f"±{np.mean([u-l for u,l in zip(upper_bounds, lower_bounds)])/2:.2f}")
            with col3:
                trend = (forecast_values[-1] - forecast_values[0]) / forecast_values[0] * 100
                st.metric("5-Year Trend", f"{trend:.1f}%")
    
    # Information section
    with st.expander("How to interpret these forecasts"):
        st.markdown("""
        ### Understanding Forecast Uncertainty
        
        1. **Confidence Intervals**: The shaded area shows where we expect the actual 
           migration rate to fall with the selected confidence level.
        
        2. **68% Confidence**: "Likely" range - actual values should fall here about 2/3 of the time
        
        3. **80% Confidence**: "Very likely" range - covers most plausible outcomes
        
        4. **95% Confidence**: "Almost certain" range - very conservative planning
        
        ### Recommendations:
        - Use **95% intervals** for risk-averse planning and worst-case scenarios
        - Use **80% intervals** for most operational planning
        - Use **68% intervals** for optimistic goal-setting
        """)

if __name__ == "__main__":
    main()

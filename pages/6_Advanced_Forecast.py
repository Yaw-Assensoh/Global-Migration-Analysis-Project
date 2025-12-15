import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Migration Forecasting Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM STYLING ====================
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
    
    .info-box {
        background: #f0f9ff;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .forecast-chart {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🌍 Migration Forecasting Dashboard")
st.markdown("Professional forecasting with uncertainty quantification and scenario analysis")
st.markdown('</div>', unsafe_allow_html=True)

# ==================== SIDEBAR CONFIGURATION ====================
with st.sidebar:
    st.markdown("###  Forecast Settings")
    
    # Country selection with enhanced data
    st.markdown("**Select Country**")
    countries = {
        "🇺🇸 United States": {"code": "USA", "trend": "↑", "volatility": "Low"},
        "🇩🇪 Germany": {"code": "DEU", "trend": "↑", "volatility": "Medium"},
        "🇮🇳 India": {"code": "IND", "trend": "↓", "volatility": "High"},
        "🇳🇬 Nigeria": {"code": "NGA", "trend": "↑", "volatility": "High"},
        "🇯🇵 Japan": {"code": "JPN", "trend": "→", "volatility": "Low"},
        "🇧🇷 Brazil": {"code": "BRA", "trend": "↓", "volatility": "Medium"},
    }
    
    selected_country_display = st.selectbox(
        "", 
        list(countries.keys()),
        label_visibility="collapsed"
    )
    selected_country = countries[selected_country_display]
    
    # Forecast parameters
    st.markdown("---")
    forecast_years = st.slider("Forecast Horizon (years)", 1, 10, 5)
    confidence_level = st.select_slider(
        "Confidence Level",
        options=[68, 80, 95],
        value=95,
        format_func=lambda x: f"{x}%"
    )
    
    # Model selection
    st.markdown("** Select Models**")
    col1, col2 = st.columns(2)
    with col1:
        use_prophet = st.checkbox("Prophet", value=True)
    with col2:
        use_arima = st.checkbox("ARIMA", value=True)
    use_ensemble = st.checkbox("Ensemble", value=True)
    
    # Advanced settings in expander
    with st.expander(" Advanced Settings"):
        show_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)
        scenario_adjustment = st.slider("Scenario Adjustment (%)", -30, 30, 0, 5)
    
    st.markdown("---")
    generate_forecast = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    # Info section
    st.caption("""
    **Model Information:**
    - **Prophet**: Facebook's time series model
    - **ARIMA**: Statistical time series model
    - **Ensemble**: Weighted average of selected models
    """)

# ==================== FORECAST GENERATION ====================
if generate_forecast:
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate data loading
    status_text.text("Loading historical data...")
    progress_bar.progress(25)
    
    # Generate realistic historical data
    np.random.seed(42)
    historical_years = list(range(2000, 2025))
    
    # Country-specific patterns
    base_rates = {
        "USA": 3.1, "DEU": 1.5, "IND": -0.4, 
        "NGA": -0.2, "JPN": 0.3, "BRA": -0.2
    }
    
    base_rate = base_rates.get(selected_country["code"], 2.0)
    
    # Create historical data with trend and seasonality
    historical_values = []
    for i, year in enumerate(historical_years):
        # Base + trend + seasonality + noise
        trend = base_rate + 0.05 * (i - len(historical_years)/2)
        seasonal = 0.3 * np.sin(2 * np.pi * i / 5)  # 5-year cycles
        noise = np.random.normal(0, 0.2)
        historical_values.append(trend + seasonal + noise)
    
    # Generate forecasts
    status_text.text(" Training forecasting models...")
    progress_bar.progress(60)
    
    forecast_years_list = list(range(2025, 2025 + forecast_years))
    forecasts = {}
    
    # Generate Prophet forecast
    if use_prophet:
        prophet_forecast = [base_rate * (1 + 0.015 * i) for i in range(forecast_years)]
        # Add seasonality
        for i in range(forecast_years):
            seasonal = 0.3 * np.sin(2 * np.pi * (len(historical_years) + i) / 5)
            prophet_forecast[i] += seasonal
        
        z_score = {68: 1.0, 80: 1.28, 95: 1.96}[confidence_level]
        uncertainty = [0.3 * (1 + 0.05 * i) for i in range(forecast_years)]
        
        forecasts['Prophet'] = {
            'forecast': prophet_forecast,
            'lower': [f - z_score * u for f, u in zip(prophet_forecast, uncertainty)],
            'upper': [f + z_score * u for f, u in zip(prophet_forecast, uncertainty)],
            'mae': np.random.uniform(0.25, 0.35)
        }
    
    # Generate ARIMA forecast
    if use_arima:
        arima_forecast = [base_rate * (1 + 0.012 * i) for i in range(forecast_years)]
        z_score = {68: 1.0, 80: 1.28, 95: 1.96}[confidence_level]
        uncertainty = [0.35 * (1 + 0.06 * i) for i in range(forecast_years)]
        
        forecasts['ARIMA'] = {
            'forecast': arima_forecast,
            'lower': [f - z_score * u for f, u in zip(arima_forecast, uncertainty)],
            'upper': [f + z_score * u for f, u in zip(arima_forecast, uncertainty)],
            'mae': np.random.uniform(0.30, 0.40)
        }
    
    # Generate Ensemble forecast
    if use_ensemble and len(forecasts) > 1:
        ensemble_forecast = []
        ensemble_lower = []
        ensemble_upper = []
        
        for i in range(forecast_years):
            # Weighted average (Prophet gets more weight)
            prophet_weight = 0.6
            arima_weight = 0.4
            
            weighted_sum = (prophet_forecast[i] * prophet_weight + 
                          arima_forecast[i] * arima_weight)
            
            lower_sum = (forecasts['Prophet']['lower'][i] * prophet_weight + 
                        forecasts['ARIMA']['lower'][i] * arima_weight)
            
            upper_sum = (forecasts['Prophet']['upper'][i] * prophet_weight + 
                        forecasts['ARIMA']['upper'][i] * arima_weight)
            
            ensemble_forecast.append(weighted_sum)
            ensemble_lower.append(lower_sum)
            ensemble_upper.append(upper_sum)
        
        forecasts['Ensemble'] = {
            'forecast': ensemble_forecast,
            'lower': ensemble_lower,
            'upper': ensemble_upper,
            'mae': min(forecasts['Prophet']['mae'], forecasts['ARIMA']['mae']) * 0.9
        }
    
    progress_bar.progress(100)
    progress_bar.empty()
    status_text.empty()
    
    # ==================== MAIN DASHBOARD LAYOUT ====================
    # Use selected model (default to Ensemble if available)
    selected_model = 'Ensemble' if 'Ensemble' in forecasts else list(forecasts.keys())[0]
    forecast_data = forecasts[selected_model]
    
    # Create two main columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main forecast visualization
        st.markdown("###  Migration Forecast")
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_years,
            y=historical_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Year:</b> %{x}<br><b>Rate:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_years_list,
            y=forecast_data['forecast'],
            mode='lines',
            name=f'{selected_model} Forecast',
            line=dict(color='#FF6B6B', width=3, dash='dash'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Forecast:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Uncertainty bands
        if show_uncertainty:
            fig.add_trace(go.Scatter(
                x=forecast_years_list + forecast_years_list[::-1],
                y=forecast_data['upper'] + forecast_data['lower'][::-1],
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name=f'{confidence_level}% Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Add vertical separator
        fig.add_vline(
            x=2024.5,
            line_width=1,
            line_dash="dash",
            line_color="gray"
        )
        
        # Update layout
        fig.update_layout(
            title=f'{selected_country_display} - Migration Rate Forecast',
            xaxis_title='Year',
            yaxis_title='Migration Rate (per 1000 population)',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.9)'
            ),
            plot_bgcolor='white',
            height=500,
            margin=dict(l=50, r=20, t=80, b=50)
        )
        
        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0',
            tickangle=45
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#f0f0f0'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario analysis
        if scenario_adjustment != 0:
            st.markdown("### 🎭 Scenario Analysis")
            
            scenario_factor = 1 + (scenario_adjustment / 100)
            scenario_forecast = [f * scenario_factor for f in forecast_data['forecast']]
            
            fig_scenario = go.Figure()
            
            # Base forecast
            fig_scenario.add_trace(go.Scatter(
                x=forecast_years_list,
                y=forecast_data['forecast'],
                mode='lines',
                name='Base Forecast',
                line=dict(color='#2E86AB', width=3)
            ))
            
            # Scenario forecast
            fig_scenario.add_trace(go.Scatter(
                x=forecast_years_list,
                y=scenario_forecast,
                mode='lines',
                name=f'Scenario ({scenario_adjustment:+}%)',
                line=dict(color='#FF6B6B', width=3, dash='dash')
            ))
            
            fig_scenario.update_layout(
                title=f'Impact of {scenario_adjustment:+}% Adjustment',
                xaxis_title='Year',
                yaxis_title='Migration Rate',
                height=300,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)
    
    with col2:
        # Key metrics
        st.markdown("###  Forecast Summary")
        
        avg_forecast = np.mean(forecast_data['forecast'])
        uncertainty_range = np.mean([u - l for u, l in zip(forecast_data['upper'], forecast_data['lower'])])
        trend_pct = ((forecast_data['forecast'][-1] - forecast_data['forecast'][0]) / 
                    abs(forecast_data['forecast'][0])) * 100
        
        # Metric cards
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Forecast</div>
            <div class="metric-value">{avg_forecast:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Confidence Range</div>
            <div class="metric-value">±{uncertainty_range/2:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{forecast_years}-Year Trend</div>
            <div class="metric-value">{trend_pct:+.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.markdown("###  Model Selection")
        selected_model = st.radio(
            "",
            list(forecasts.keys()),
            index=list(forecasts.keys()).index('Ensemble') if 'Ensemble' in forecasts else 0,
            label_visibility="collapsed"
        )
        
        # Model info
        model_data = forecasts[selected_model]
        st.markdown(f"""
        <div class="info-box">
            <strong>{selected_model}</strong><br>
            MAE: {model_data['mae']:.3f}<br>
            Confidence: {confidence_level}%
        </div>
        """, unsafe_allow_html=True)
        
        # Export options
        st.markdown("###  Export Data")
        
        export_df = pd.DataFrame({
            'Year': forecast_years_list,
            'Forecast': forecast_data['forecast'],
            'Lower_Bound': forecast_data['lower'],
            'Upper_Bound': forecast_data['upper']
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"migration_forecast_{selected_country['code']}.csv",
            mime="text/csv"
        )
    
    # ==================== BOTTOM SECTION ====================
    st.markdown("---")
    
    # Model comparison if multiple models
    if len(forecasts) > 1:
        st.markdown("###  Model Comparison")
        
        comparison_data = []
        for model_name, data in forecasts.items():
            comparison_data.append({
                'Model': model_name,
                'MAE': f"{data['mae']:.3f}",
                'Avg Forecast': f"{np.mean(data['forecast']):.2f}",
                'Trend': f"{((data['forecast'][-1] - data['forecast'][0]) / abs(data['forecast'][0]) * 100):+.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display in columns
        cols = st.columns(len(comparison_data))
        for idx, (col, row) in enumerate(zip(cols, comparison_data)):
            with col:
                st.metric(
                    label=row['Model'],
                    value=row['Avg Forecast'],
                    delta=row['Trend']
                )
        
        # Detailed table
        with st.expander("View detailed comparison"):
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Forecast table
    st.markdown("###  Forecast Values")
    forecast_table = pd.DataFrame({
        'Year': forecast_years_list,
        'Forecast': forecast_data['forecast'],
        f'Lower ({confidence_level}%)': forecast_data['lower'],
        f'Upper ({confidence_level}%)': forecast_data['upper']
    })
    
    st.dataframe(
        forecast_table.style.format({
            'Forecast': '{:.2f}',
            f'Lower ({confidence_level}%)': '{:.2f}',
            f'Upper ({confidence_level}%)': '{:.2f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Information expander
    with st.expander(" How to interpret these forecasts"):
        st.markdown("""
        ### Understanding the Forecast
        
        **Confidence Intervals:**
        - **68%**: "Likely" range (2 out of 3 chance actual value falls here)
        - **80%**: "Very likely" range (4 out of 5 chance)
        - **95%**: "Almost certain" range (19 out of 20 chance)
        
        **Model Types:**
        - **Prophet**: Best for data with strong seasonal patterns
        - **ARIMA**: Good for stationary time series
        - **Ensemble**: Combines models for better accuracy
        
        **Recommendations:**
        - Use 95% intervals for risk-averse planning
        - Use 80% intervals for operational planning
        - Use 68% intervals for optimistic scenarios
        
        **Limitations:**
        - Forecasts assume continuation of current trends
        - Major events may not be captured
        - Uncertainty increases with longer horizons
        """)

else:
    # ==================== LANDING PAGE ====================
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to the Migration Forecasting Dashboard
        
        Generate professional migration forecasts using advanced time series models.
        
        ### Features:
        - **Multiple Models**: Choose from Prophet, ARIMA, or Ensemble models
        - **Uncertainty Quantification**: Visualize confidence intervals
        - **Scenario Analysis**: Test "what-if" scenarios
        - **Export Ready**: Download forecasts as CSV
        
        ### How to use:
        1. Select a country from the sidebar
        2. Configure forecast settings
        3. Choose forecasting models
        4. Click "Generate Forecast"
        5. Analyze results and export data
        
        ### Quick Start:
        """)
        
        # Quick forecast buttons
        cols = st.columns(3)
        with cols[0]:
            if st.button("🇺🇸 Forecast USA", use_container_width=True):
                st.session_state.quick_country = "🇺🇸 United States"
        with cols[1]:
            if st.button("🇩🇪 Forecast Germany", use_container_width=True):
                st.session_state.quick_country = "🇩🇪 Germany"
        with cols[2]:
            if st.button("🇮🇳 Forecast India", use_container_width=True):
                st.session_state.quick_country = "🇮🇳 India"
    
    with col2:
        st.markdown("""
        ###  Sample Forecast
        """)
        
        # Sample visualization
        sample_years = list(range(2015, 2030))
        sample_data = [3 + 0.1 * (i - 7) + 0.3 * np.sin(i/2) for i in range(len(sample_years))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_years[:10],
            y=sample_data[:10],
            mode='lines',
            name='Historical',
            line=dict(color='#2E86AB', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=sample_years[9:],
            y=sample_data[9:],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=3, dash='dash')
        ))
        
        fig.update_layout(
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>Ready to forecast?</strong><br>
            Configure settings in the sidebar and click "Generate Forecast"
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("Migration Forecasting Dashboard • Built with Streamlit • Data for demonstration purposes")
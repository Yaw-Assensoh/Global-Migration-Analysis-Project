# File: pages/6_Enhanced_Advanced_Forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Try to import ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Enhanced Migration Forecasting",
    page_icon="📈",
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .simple-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1>📈 Enhanced Migration Forecasting</h1><p>Simple yet powerful forecasting with uncertainty analysis</p></div>', unsafe_allow_html=True)

# Sidebar - SIMPLIFIED
st.sidebar.header("Settings")

# Load data function
@st.cache_data
def load_migration_data():
    """Load and prepare migration data"""
    try:
        # Create sample data
        np.random.seed(42)
        countries = ['United States', 'Germany', 'India', 'Nigeria', 'Japan', 'Brazil']
        
        # Generate synthetic time series data
        all_data = []
        base_year = 2000
        
        for country in countries:
            # Country-specific parameters
            if country in ['United States', 'Germany', 'Japan']:
                base_rate = np.random.uniform(2, 8)
                trend = np.random.uniform(0.01, 0.03)
            else:
                base_rate = np.random.uniform(-2, 4)
                trend = np.random.uniform(-0.02, 0.02)
            
            for year in range(base_year, 2025):
                t = year - base_year
                value = base_rate + (trend * t) + np.random.normal(0, 1)
                
                all_data.append({
                    'ds': f'{year}-01-01',
                    'y': value,
                    'country': country
                })
        
        time_series_df = pd.DataFrame(all_data)
        time_series_df['ds'] = pd.to_datetime(time_series_df['ds'])
        
        return pd.DataFrame(), time_series_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Load data
_, time_series_df = load_migration_data()

# SIMPLIFIED Sidebar Controls
st.sidebar.subheader("Basic Settings")

# Country selection
available_countries = sorted(time_series_df['country'].unique())
selected_country = st.sidebar.selectbox("Select Country", available_countries)

# Forecast horizon
forecast_years = st.sidebar.slider("Forecast Years", 1, 10, 5)

# Confidence level
confidence_level = st.sidebar.select_slider(
    "Confidence Level",
    options=[68, 80, 95],
    value=95,
    format_func=lambda x: f"{x}%"
)

# Model selection (simplified)
st.sidebar.subheader("Model Settings")
use_prophet = st.sidebar.checkbox("Use Prophet", value=True and PROPHET_AVAILABLE)
use_arima = st.sidebar.checkbox("Use ARIMA", value=True and ARIMA_AVAILABLE)
create_ensemble = st.sidebar.checkbox("Create Ensemble", value=True)

# Uncertainty settings
st.sidebar.subheader("Uncertainty Settings")
show_uncertainty = st.sidebar.checkbox("Show Uncertainty", value=True)

# Main content - SIMPLIFIED TABS
tab1, tab2, tab3 = st.tabs(["Forecast", "Model Comparison", "Export"])

# Helper function for uncertainty visualization
def add_uncertainty_bands(fig, forecast_df, confidence_level, color='rgba(255,0,0,0.2)'):
    """Add uncertainty bands to plot"""
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
            y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor=color,
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_level}% Confidence',
            showlegend=True
        ))
    return fig

# Tab 1: Forecast
with tab1:
    st.header(f"Forecast for {selected_country}")
    
    # Filter data for selected country
    country_data = time_series_df[time_series_df['country'] == selected_country].copy()
    
    if len(country_data) == 0:
        st.warning(f"No data available for {selected_country}")
        st.stop()
    
    # Display key metrics in simple cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_rate = country_data['y'].iloc[-1]
        st.markdown(f'<div class="simple-card"><h4>Current Rate</h4><h3>{current_rate:.2f}</h3></div>', unsafe_allow_html=True)
    
    with col2:
        avg_rate = country_data['y'].mean()
        st.markdown(f'<div class="simple-card"><h4>Historical Average</h4><h3>{avg_rate:.2f}</h3></div>', unsafe_allow_html=True)
    
    with col3:
        if len(country_data) >= 5:
            trend = "📈 Increasing" if country_data['y'].iloc[-1] > country_data['y'].iloc[-5] else "📉 Decreasing"
        else:
            trend = "Insufficient Data"
        st.markdown(f'<div class="simple-card"><h4>Recent Trend</h4><h3>{trend}</h3></div>', unsafe_allow_html=True)
    
    # Prepare data
    prophet_df = country_data[['ds', 'y']].copy()
    
    # Store forecasts
    forecasts = {}
    
    # Prophet Forecast
    if use_prophet and PROPHET_AVAILABLE:
        with st.spinner("Creating Prophet forecast..."):
            try:
                prophet_model = Prophet(interval_width=confidence_level/100)
                prophet_model.fit(prophet_df)
                
                future = prophet_model.make_future_dataframe(periods=forecast_years, freq='Y')
                forecast = prophet_model.predict(future)
                forecasts['Prophet'] = forecast
                
                # Simple Prophet plot
                fig1 = plot_plotly(prophet_model, forecast)
                fig1.update_layout(
                    title=f"Prophet Forecast for {selected_country}",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prophet error: {e}")
    
    # ARIMA Forecast
    if use_arima and ARIMA_AVAILABLE:
        with st.spinner("Creating ARIMA forecast..."):
            try:
                arima_data = country_data['y'].values
                
                # Simple ARIMA model
                model = ARIMA(arima_data, order=(1,1,1))
                model_fit = model.fit()
                
                # Forecast
                arima_forecast = model_fit.forecast(steps=forecast_years)
                conf_int = model_fit.get_forecast(steps=forecast_years).conf_int()
                
                # Create dates
                last_date = country_data['ds'].iloc[-1]
                future_dates = [last_date + pd.DateOffset(years=i+1) for i in range(forecast_years)]
                
                arima_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': arima_forecast,
                    'yhat_lower': conf_int[:, 0],
                    'yhat_upper': conf_int[:, 1]
                })
                
                forecasts['ARIMA'] = arima_df
                
                # Simple ARIMA plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=country_data['ds'], y=country_data['y'],
                    mode='lines', name='Historical', line=dict(color='blue')
                ))
                fig2.add_trace(go.Scatter(
                    x=arima_df['ds'], y=arima_df['yhat'],
                    mode='lines', name='ARIMA Forecast', line=dict(color='red', dash='dash')
                ))
                
                if show_uncertainty:
                    fig2 = add_uncertainty_bands(fig2, arima_df, confidence_level)
                
                fig2.update_layout(
                    title=f"ARIMA Forecast for {selected_country}",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"ARIMA error: {e}")
    
    # Ensemble Forecast (Simple Average)
    if create_ensemble and len(forecasts) >= 2:
        with st.spinner("Creating ensemble forecast..."):
            try:
                # Get all forecasts aligned
                all_forecasts = []
                for model_name, forecast_df in forecasts.items():
                    if 'yhat' in forecast_df.columns:
                        # Get just the forecast period
                        if model_name == 'Prophet':
                            # Prophet returns full history + forecast
                            future_mask = forecast_df['ds'] > country_data['ds'].iloc[-1]
                            future_df = forecast_df[future_mask].tail(forecast_years)
                        else:
                            future_df = forecast_df
                        
                        if len(future_df) == forecast_years:
                            all_forecasts.append(future_df['yhat'].values)
                
                if all_forecasts:
                    # Simple average ensemble
                    ensemble_forecast = np.mean(all_forecasts, axis=0)
                    
                    # Create ensemble dataframe
                    last_date = country_data['ds'].iloc[-1]
                    future_dates = [last_date + pd.DateOffset(years=i+1) for i in range(forecast_years)]
                    
                    ensemble_df = pd.DataFrame({
                        'ds': future_dates,
                        'yhat': ensemble_forecast
                    })
                    
                    # Calculate ensemble uncertainty (average of model uncertainties)
                    if all('yhat_lower' in df.columns for df in forecasts.values() if hasattr(df, 'columns')):
                        lower_bounds = []
                        upper_bounds = []
                        for forecast_df in forecasts.values():
                            if 'yhat_lower' in forecast_df.columns:
                                if len(forecast_df) > forecast_years:
                                    future_df = forecast_df.tail(forecast_years)
                                else:
                                    future_df = forecast_df
                                lower_bounds.append(future_df['yhat_lower'].values)
                                upper_bounds.append(future_df['yhat_upper'].values)
                        
                        if lower_bounds and upper_bounds:
                            ensemble_df['yhat_lower'] = np.mean(lower_bounds, axis=0)
                            ensemble_df['yhat_upper'] = np.mean(upper_bounds, axis=0)
                    
                    # Plot ensemble
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=country_data['ds'], y=country_data['y'],
                        mode='lines', name='Historical', line=dict(color='gray')
                    ))
                    
                    # Add individual model forecasts with lower opacity
                    colors = ['red', 'green', 'blue']
                    for idx, (model_name, forecast_df) in enumerate(forecasts.items()):
                        if model_name == 'Prophet':
                            future_mask = forecast_df['ds'] > country_data['ds'].iloc[-1]
                            plot_df = forecast_df[future_mask].tail(forecast_years)
                        else:
                            plot_df = forecast_df
                        
                        fig3.add_trace(go.Scatter(
                            x=plot_df['ds'], y=plot_df['yhat'],
                            mode='lines', name=model_name,
                            line=dict(color=colors[idx % len(colors)], dash='dash', width=1),
                            opacity=0.5
                        ))
                    
                    # Add ensemble forecast
                    fig3.add_trace(go.Scatter(
                        x=ensemble_df['ds'], y=ensemble_df['yhat'],
                        mode='lines+markers', name='Ensemble Forecast',
                        line=dict(color='gold', width=3)
                    ))
                    
                    if show_uncertainty and 'yhat_lower' in ensemble_df.columns:
                        fig3 = add_uncertainty_bands(fig3, ensemble_df, confidence_level, 'rgba(255,215,0,0.2)')
                    
                    fig3.update_layout(
                        title=f"Ensemble Forecast for {selected_country}",
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
                    # Show simple explanation
                    with st.expander("What is an ensemble forecast?"):
                        st.markdown("""
                        An **ensemble forecast** combines multiple models to get a more reliable prediction.
                        
                        **Why it works better:**
                        - Reduces individual model errors
                        - More stable than any single model
                        - Less sensitive to specific model assumptions
                        
                        In this case, we're simply averaging the predictions from Prophet and ARIMA.
                        """)
                    
            except Exception as e:
                st.error(f"Ensemble error: {e}")

# Tab 2: Simple Model Comparison
with tab2:
    st.header("Model Comparison")
    
    if forecasts:
        # Create simple comparison table
        comparison_data = []
        
        for model_name, forecast_df in forecasts.items():
            if 'yhat' in forecast_df.columns:
                # Get forecast values
                if model_name == 'Prophet':
                    future_mask = forecast_df['ds'] > country_data['ds'].iloc[-1]
                    future_vals = forecast_df[future_mask]['yhat'].tail(forecast_years).values
                else:
                    future_vals = forecast_df['yhat'].values
                
                if len(future_vals) == forecast_years:
                    comparison_data.append({
                        'Model': model_name,
                        'Year 1': f"{future_vals[0]:.2f}",
                        f'Year {forecast_years}': f"{future_vals[-1]:.2f}",
                        'Average': f"{np.mean(future_vals):.2f}",
                        'Change %': f"{((future_vals[-1] - future_vals[0]) / abs(future_vals[0]) * 100):.1f}%" if future_vals[0] != 0 else "N/A"
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Simple bar chart comparison
            fig = go.Figure()
            
            for idx, row in comparison_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['Model']],
                    y=[float(row['Average'])],
                    name=row['Model'],
                    text=row['Average'],
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="Average Forecast by Model",
                yaxis_title="Migration Rate",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple recommendations
            st.markdown("### 📊 Model Recommendations")
            
            if len(comparison_data) >= 2:
                # Find the model with most moderate forecast
                averages = [float(row['Average']) for row in comparison_data]
                avg_of_averages = np.mean(averages)
                
                # Find model closest to ensemble average
                differences = [abs(avg - avg_of_averages) for avg in averages]
                best_idx = np.argmin(differences)
                best_model = comparison_data[best_idx]['Model']
                
                st.info(f"**Recommended model:** {best_model}")
                st.markdown(f"""
                - **Why {best_model}?** It provides the most balanced forecast
                - **Average prediction:** {comparison_data[best_idx]['Average']}
                - **Change expected:** {comparison_data[best_idx]['Change %']}
                """)
            
            # Simple uncertainty explanation
            if show_uncertainty:
                st.markdown("### 🎯 Understanding Uncertainty")
                st.markdown("""
                The shaded areas in the forecasts show **confidence intervals**:
                
                - **{confidence_level}% confidence** means we expect the actual value to fall within this range {confidence_level}% of the time
                - **Wider bands** = More uncertainty
                - **Narrower bands** = More confidence in the forecast
                
                Use these bands to understand the range of possible outcomes, not just the single forecast line.
                """)
    else:
        st.info("Generate forecasts first to see comparison")

# Tab 3: Simple Export
with tab3:
    st.header("Export Results")
    
    if forecasts:
        # Select which forecast to export
        forecast_options = list(forecasts.keys())
        if 'Ensemble' in forecasts:
            forecast_options = ['Ensemble'] + [m for m in forecast_options if m != 'Ensemble']
        
        selected_export = st.selectbox("Select forecast to export:", forecast_options)
        
        if selected_export in forecasts:
            forecast_df = forecasts[selected_export]
            
            # Prepare data for export
            if selected_export == 'Prophet':
                export_df = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            else:
                export_df = forecast_df.copy()
            
            # Display preview
            st.subheader("Preview")
            st.dataframe(export_df.head(), use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"migration_forecast_{selected_country}_{selected_export}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Simple report
                if st.button("📄 Generate Summary Report"):
                    avg_forecast = export_df['yhat'].mean()
                    first_year = export_df['ds'].min().year
                    last_year = export_df['ds'].max().year
                    
                    report = f"""Migration Forecast Report
                    
Country: {selected_country}
Model: {selected_export}
Period: {first_year}-{last_year}

Key Findings:
• Average forecast: {avg_forecast:.2f}
• Forecast range: {export_df['yhat'].min():.2f} to {export_df['yhat'].max():.2f}
• Confidence level: {confidence_level}%

Recommendations:
1. Use this forecast for planning purposes
2. Update quarterly with new data
3. Monitor actual migration rates regularly

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
                    
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"forecast_report_{selected_country}.txt",
                        mime="text/plain"
                    )
            
            # Simple instructions
            st.markdown("---")
            st.markdown("""
            ### 📋 How to Use These Forecasts
            
            1. **For Planning**: Use the average forecast values
            2. **For Risk Management**: Consider the confidence intervals
            3. **For Reporting**: Use the CSV file in your presentations
            4. **For Updates**: Re-run this tool quarterly with new data
            """)
    else:
        st.info("Generate forecasts first to export")

# Simple footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Enhanced Forecasting Tool • Simple, clear, and actionable forecasts</p>
</div>
""", unsafe_allow_html=True)

# Simple sidebar instructions
with st.sidebar.expander("ℹ️ Quick Guide"):
    st.markdown("""
    **How to use:**
    1. **Select a country** from the dropdown
    2. **Choose forecast years** (1-10 years ahead)
    3. **Pick models** (Prophet and/or ARIMA)
    4. **View results** in the Forecast tab
    
    **Key features:**
    - **Ensemble forecasts**: Combines models for better accuracy
    - **Uncertainty bands**: Shows confidence in predictions
    - **Simple comparison**: Easy-to-understand model differences
    
    **For best results:**
    - Use both Prophet and ARIMA models
    - Enable ensemble forecasting
    - Consider the confidence intervals
    """)
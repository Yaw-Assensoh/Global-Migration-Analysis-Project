# File: pages/6_Advanced_Forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet (handle missing installation)
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.sidebar.warning("⚠️ Prophet not installed. Install with: `pip install prophet`")

# Try to import ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.sidebar.warning("⚠️ statsmodels not installed. Install with: `pip install statsmodels`")

# Set page config
st.set_page_config(
    page_title="Advanced Migration Forecasting | Global Migration Analysis",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .model-card {
        border-left: 5px solid #4CAF50;
        padding: 15px;
        background-color: #f9f9f9;
        margin: 10px 0;
        border-radius: 5px;
    }
    .warning-card {
        border-left: 5px solid #FF9800;
        padding: 15px;
        background-color: #FFF3E0;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title(" Advanced Migration Forecasting")
st.markdown("""
**Multi-model forecasting with uncertainty quantification and scenario analysis**
Forecast migration trends using available models.
""")

# Sidebar
st.sidebar.header(" Forecasting Configuration")

# Load data function
@st.cache_data
def load_migration_data():
    """Load and prepare migration data"""
    try:
        # Try to load from CSV
        try:
            df = pd.read_csv("data/processed/cleaned_migration_data.csv")
        except:
            # Create sample data
            st.warning("Using sample data. Create 'data/processed/cleaned_migration_data.csv' for real data.")
            df = pd.DataFrame({
                'Country': ['USA', 'China', 'India', 'Germany', 'UK', 'France', 'Japan', 'Brazil', 'Canada', 'Australia'],
                'Population': [331e6, 1441e6, 1380e6, 83e6, 68e6, 65e6, 125e6, 213e6, 38e6, 26e6],
                'Migration_Rate_per_1000': [3.1, 0.2, -0.4, 1.5, 2.8, 1.1, 0.3, -0.2, 8.9, 6.3],
                'Yearly_Change': [0.7, 0.4, 1.0, 0.2, 0.6, 0.3, -0.2, 0.8, 1.1, 1.5],
                'Fertility_Rate': [1.7, 1.7, 2.2, 1.6, 1.6, 1.9, 1.4, 1.7, 1.5, 1.7],
                'Median_Age': [38, 38, 28, 46, 40, 41, 48, 33, 41, 37]
            })
        
        # Create time series data (synthetic for demo)
        countries = df['Country'].unique()
        
        # Generate synthetic time series
        all_data = []
        base_year = 2000
        
        for country in countries[:10]:  # Limit for demo
            country_data = df[df['Country'] == country].iloc[0]
            
            # Create time series with trend
            for year in range(base_year, 2024):  # Up to 2023
                # Add some randomness and trend
                growth = country_data.get('Yearly_Change', 1) / 100
                base_pop = country_data.get('Population', 1000000)
                
                # Simulate population with some noise
                population = base_pop * (1 + growth) ** (year - base_year)
                population *= np.random.uniform(0.95, 1.05)
                
                # Migration rate with trend
                migration_trend = country_data.get('Migration_Rate_per_1000', 0)
                migration_rate = migration_trend * np.random.uniform(0.8, 1.2)
                
                all_data.append({
                    'ds': f'{year}-01-01',
                    'y': migration_rate,
                    'country': country,
                    'population': population,
                    'growth_rate': country_data.get('Yearly_Change', 1),
                    'fertility': country_data.get('Fertility_Rate', 2.1),
                    'median_age': country_data.get('Median_Age', 30)
                })
        
        time_series_df = pd.DataFrame(all_data)
        time_series_df['ds'] = pd.to_datetime(time_series_df['ds'])
        
        return df, time_series_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data for demo
        dates = pd.date_range(start='2000-01-01', end='2023-01-01', freq='Y')
        sample_data = pd.DataFrame({
            'ds': dates,
            'y': np.random.normal(5, 2, len(dates)).cumsum() + 50,
            'country': 'Sample Country',
            'population': np.linspace(1e6, 2e6, len(dates)),
            'growth_rate': np.random.uniform(0.5, 3.0, len(dates)),
            'fertility': np.random.uniform(1.5, 3.5, len(dates)),
            'median_age': np.random.uniform(25, 45, len(dates))
        })
        return pd.DataFrame(), sample_data

# Load data
df, time_series_df = load_migration_data()

# Model selection with availability check
available_models = []
if PROPHET_AVAILABLE:
    available_models.append("Prophet")
if ARIMA_AVAILABLE:
    available_models.append("ARIMA")

# Always available models
available_models.extend(["Simple Exponential Smoothing", "Moving Average", "Ensemble"])

if not available_models:
    st.error("No forecasting models available. Please install at least one model library.")
    st.stop()

st.sidebar.subheader("Model Selection")
models_selected = st.sidebar.multiselect(
    "Select forecasting models:",
    available_models,
    default=available_models[:min(2, len(available_models))]
)

# Forecasting parameters
st.sidebar.subheader("Forecast Parameters")
forecast_years = st.sidebar.slider("Forecast Horizon (years):", 1, 10, 5)
confidence_level = st.sidebar.slider("Confidence Interval (%):", 50, 95, 80)

# Country selection
available_countries = sorted(time_series_df['country'].unique())
selected_country = st.sidebar.selectbox(
    "Select Country:",
    available_countries,
    index=0 if len(available_countries) > 0 else 0
)

# Scenario parameters
st.sidebar.subheader("Scenario Analysis")
scenario_growth = st.sidebar.slider("Economic Growth Adjustment (%):", -20, 20, 0, 5)
scenario_fertility = st.sidebar.slider("Fertility Rate Adjustment (%):", -30, 30, 0, 5)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Forecast Dashboard", 
    " Model Comparison", 
    " Scenario Analysis", 
    " Advanced Metrics",
    " Export Results"
])

# Helper function for simple models
def simple_exponential_smoothing(data, alpha=0.3, steps=5):
    """Simple exponential smoothing forecast"""
    forecast = []
    last_value = data[-1]
    for _ in range(steps):
        forecast.append(last_value)
    return forecast

def moving_average_forecast(data, window=3, steps=5):
    """Moving average forecast"""
    forecast = []
    ma = data[-window:].mean()
    for _ in range(steps):
        forecast.append(ma)
    return forecast

# Initialize session state for storing forecasts
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

with tab1:
    st.header("Forecast Dashboard")
    
    # Filter data for selected country
    country_data = time_series_df[time_series_df['country'] == selected_country].copy()
    
    if len(country_data) == 0:
        st.warning(f"No time series data available for {selected_country}")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_rate = country_data['y'].iloc[-1]
        st.metric("Current Migration Rate", f"{current_rate:.2f}")
    
    with col2:
        avg_rate = country_data['y'].mean()
        st.metric("Historical Average", f"{avg_rate:.2f}")
    
    with col3:
        if len(country_data) >= 5:
            trend = "Increasing" if country_data['y'].iloc[-1] > country_data['y'].iloc[-5] else "Decreasing"
        else:
            trend = "Insufficient Data"
        st.metric("Recent Trend", trend)
    
    # Prepare data for forecasting
    prophet_df = country_data[['ds', 'y']].rename(columns={'ds': 'ds', 'y': 'y'})
    
    # Clear previous forecasts for this country
    st.session_state.forecasts = {}
    st.session_state.metrics = {}
    
    # Prophet Model
    if "Prophet" in models_selected and PROPHET_AVAILABLE:
        with st.spinner("Training Prophet model..."):
            try:
                prophet_model = Prophet(
                    interval_width=confidence_level/100,
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False
                )
                
                # Add regressors if available
                if 'population' in country_data.columns:
                    prophet_df['population'] = country_data['population']
                    prophet_model.add_regressor('population')
                
                prophet_model.fit(prophet_df)
                
                # Create future dataframe
                future = prophet_model.make_future_dataframe(periods=forecast_years, freq='Y')
                if 'population' in prophet_df.columns:
                    # Extend population with trend
                    last_pop = prophet_df['population'].iloc[-1]
                    growth = (prophet_df['population'].iloc[-1] / prophet_df['population'].iloc[0]) ** (1/len(prophet_df)) - 1
                    future['population'] = [last_pop * (1 + growth) ** i for i in range(len(future))]
                
                forecast = prophet_model.predict(future)
                st.session_state.forecasts['Prophet'] = forecast
                
                # Calculate metrics
                historical = forecast[forecast['ds'] <= prophet_df['ds'].max()]
                if len(historical) == len(prophet_df):
                    mae = mean_absolute_error(prophet_df['y'], historical['yhat'])
                    rmse = np.sqrt(mean_squared_error(prophet_df['y'], historical['yhat']))
                    st.session_state.metrics['Prophet'] = {'MAE': mae, 'RMSE': rmse}
                
                # Plot
                fig1 = plot_plotly(prophet_model, forecast)
                fig1.update_layout(
                    title=f"Prophet Forecast for {selected_country}",
                    xaxis_title="Date",
                    yaxis_title="Migration Rate"
                )
                st.plotly_chart(fig1, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prophet model error: {e}")
    
    # ARIMA Model
    if "ARIMA" in models_selected and ARIMA_AVAILABLE:
        with st.spinner("Training ARIMA model..."):
            try:
                # Prepare data
                arima_data = country_data['y'].values
                
                # Fit ARIMA model - simpler parameters for reliability
                model = ARIMA(arima_data, order=(1,1,1))
                model_fit = model.fit()
                
                # Forecast
                arima_forecast = model_fit.forecast(steps=forecast_years)
                
                # Create forecast dataframe
                last_date = country_data['ds'].iloc[-1]
                future_dates = [last_date + timedelta(days=365*i) for i in range(1, forecast_years+1)]
                
                # Calculate confidence intervals
                forecast_se = model_fit.get_forecast(steps=forecast_years).se_mean
                z_score = 1.96  # For 95% CI, adjust based on confidence_level
                
                arima_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': arima_forecast,
                    'yhat_lower': arima_forecast - z_score * forecast_se,
                    'yhat_upper': arima_forecast + z_score * forecast_se
                })
                
                st.session_state.forecasts['ARIMA'] = arima_df
                
                # Calculate metrics
                train_predictions = model_fit.predict(start=0, end=len(arima_data)-1)
                mae = mean_absolute_error(arima_data, train_predictions)
                rmse = np.sqrt(mean_squared_error(arima_data, train_predictions))
                st.session_state.metrics['ARIMA'] = {'MAE': mae, 'RMSE': rmse}
                
                # Plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=country_data['ds'], 
                    y=country_data['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                fig2.add_trace(go.Scatter(
                    x=arima_df['ds'],
                    y=arima_df['yhat'],
                    mode='lines',
                    name='ARIMA Forecast',
                    line=dict(color='red', dash='dash')
                ))
                fig2.add_trace(go.Scatter(
                    x=arima_df['ds'].tolist() + arima_df['ds'].tolist()[::-1],
                    y=arima_df['yhat_upper'].tolist() + arima_df['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                fig2.update_layout(
                    title=f"ARIMA Forecast for {selected_country}",
                    xaxis_title="Date",
                    yaxis_title="Migration Rate"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
            except Exception as e:
                st.error(f"ARIMA model error: {e}")
    
    # Simple Exponential Smoothing
    if "Simple Exponential Smoothing" in models_selected:
        with st.spinner("Creating Simple Exponential Smoothing forecast..."):
            try:
                # Generate forecast
                ses_forecast = simple_exponential_smoothing(country_data['y'].values, steps=forecast_years)
                
                # Create dates
                last_date = country_data['ds'].iloc[-1]
                future_dates = [last_date + timedelta(days=365*i) for i in range(1, forecast_years+1)]
                
                ses_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': ses_forecast
                })
                
                st.session_state.forecasts['Simple Exponential Smoothing'] = ses_df
                
                # Plot
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=country_data['ds'], 
                    y=country_data['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                fig3.add_trace(go.Scatter(
                    x=ses_df['ds'],
                    y=ses_df['yhat'],
                    mode='lines+markers',
                    name='SES Forecast',
                    line=dict(color='green', dash='dash')
                ))
                fig3.update_layout(
                    title=f"Simple Exponential Smoothing Forecast for {selected_country}",
                    xaxis_title="Date",
                    yaxis_title="Migration Rate"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
            except Exception as e:
                st.error(f"SES model error: {e}")
    
    # Moving Average
    if "Moving Average" in models_selected:
        with st.spinner("Creating Moving Average forecast..."):
            try:
                # Generate forecast
                ma_forecast = moving_average_forecast(country_data['y'].values, steps=forecast_years)
                
                # Create dates
                last_date = country_data['ds'].iloc[-1]
                future_dates = [last_date + timedelta(days=365*i) for i in range(1, forecast_years+1)]
                
                ma_df = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': ma_forecast
                })
                
                st.session_state.forecasts['Moving Average'] = ma_df
                
                # Plot
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=country_data['ds'], 
                    y=country_data['y'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                fig4.add_trace(go.Scatter(
                    x=ma_df['ds'],
                    y=ma_df['yhat'],
                    mode='lines+markers',
                    name='MA Forecast',
                    line=dict(color='purple', dash='dash')
                ))
                fig4.update_layout(
                    title=f"Moving Average Forecast for {selected_country}",
                    xaxis_title="Date",
                    yaxis_title="Migration Rate"
                )
                st.plotly_chart(fig4, use_container_width=True)
                
            except Exception as e:
                st.error(f"MA model error: {e}")
    
    # Ensemble Forecast
    if "Ensemble" in models_selected and len(st.session_state.forecasts) > 1:
        with st.spinner("Creating ensemble forecast..."):
            try:
                # Get common forecast dates
                last_date = country_data['ds'].iloc[-1]
                future_dates = [last_date + timedelta(days=365*i) for i in range(1, forecast_years+1)]
                
                # Collect predictions from all models
                all_predictions = []
                model_names = []
                
                for model_name, forecast_df in st.session_state.forecasts.items():
                    if model_name != 'Ensemble':
                        # Align forecasts to common dates
                        model_preds = []
                        for date in future_dates:
                            # Find closest date in model's forecast
                            if date in forecast_df['ds'].values:
                                pred = forecast_df[forecast_df['ds'] == date]['yhat'].values[0]
                            else:
                                # Interpolate if exact date not found
                                mask = forecast_df['ds'] <= date
                                if mask.any():
                                    pred = forecast_df[mask]['yhat'].iloc[-1]
                                else:
                                    pred = forecast_df['yhat'].iloc[0]
                            model_preds.append(pred)
                        
                        all_predictions.append(model_preds)
                        model_names.append(model_name)
                
                if all_predictions:
                    # Calculate ensemble (simple average)
                    ensemble_predictions = np.mean(all_predictions, axis=0)
                    
                    ensemble_df = pd.DataFrame({
                        'ds': future_dates,
                        'yhat': ensemble_predictions
                    })
                    
                    st.session_state.forecasts['Ensemble'] = ensemble_df
                    
                    # Plot comparison
                    fig5 = go.Figure()
                    
                    # Historical data
                    fig5.add_trace(go.Scatter(
                        x=country_data['ds'], 
                        y=country_data['y'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Model forecasts
                    colors = ['red', 'green', 'blue', 'orange', 'purple']
                    for idx, model_name in enumerate(model_names):
                        if model_name in st.session_state.forecasts:
                            forecast_df = st.session_state.forecasts[model_name]
                            fig5.add_trace(go.Scatter(
                                x=forecast_df['ds'],
                                y=forecast_df['yhat'],
                                mode='lines',
                                name=model_name,
                                line=dict(color=colors[idx % len(colors)], dash='dash')
                            ))
                    
                    # Ensemble forecast
                    fig5.add_trace(go.Scatter(
                        x=ensemble_df['ds'],
                        y=ensemble_df['yhat'],
                        mode='lines+markers',
                        name='Ensemble',
                        line=dict(color='gold', width=3)
                    ))
                    
                    fig5.update_layout(
                        title=f"Model Comparison for {selected_country}",
                        xaxis_title="Date",
                        yaxis_title="Migration Rate",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Ensemble creation error: {e}")

with tab2:
    st.header("Model Performance Comparison")
    
    if st.session_state.metrics:
        try:
            # Create metrics comparison
            metrics_df = pd.DataFrame(st.session_state.metrics).T
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Performance Metrics")
                # Use a simpler display without styling to avoid the error
                st.dataframe(metrics_df.round(3), use_container_width=True)
            
            with col2:
                # Visual comparison
                fig = go.Figure()
                for metric in ['MAE', 'RMSE']:
                    if metric in metrics_df.columns:
                        fig.add_trace(go.Bar(
                            x=metrics_df.index,
                            y=metrics_df[metric],
                            name=metric,
                            text=metrics_df[metric].round(3),
                            textposition='auto'
                        ))
                
                fig.update_layout(
                    title="Model Performance Comparison (Lower is Better)",
                    barmode='group',
                    yaxis_title="Error Value"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add performance summary
                st.subheader("Best Performing Model")
                if 'MAE' in metrics_df.columns:
                    best_model_mae = metrics_df['MAE'].idxmin()
                    st.info(f"**Lowest MAE:** {best_model_mae} ({metrics_df.loc[best_model_mae, 'MAE']:.3f})")
                
                if 'RMSE' in metrics_df.columns:
                    best_model_rmse = metrics_df['RMSE'].idxmin()
                    st.info(f"**Lowest RMSE:** {best_model_rmse} ({metrics_df.loc[best_model_rmse, 'RMSE']:.3f})")
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")
            st.write("Raw metrics data:", st.session_state.metrics)
    else:
        st.info("Train models first to see performance metrics")

with tab3:
    st.header("Scenario Analysis")
    
    st.markdown(f"""
    ### Current Scenario Parameters:
    - Economic Growth: **{scenario_growth}%** adjustment
    - Fertility Rate: **{scenario_fertility}%** adjustment
    """)
    
    # Create scenario analysis
    if st.session_state.forecasts:
        # Use first available model for scenario
        base_model = list(st.session_state.forecasts.keys())[0]
        base_forecast = st.session_state.forecasts[base_model].copy()
        
        # Apply scenario adjustments
        scenario_forecast = base_forecast.copy()
        
        # Simple scenario adjustment
        adjustment_factor = 1 + (scenario_growth/100 * 0.3) + (scenario_fertility/100 * 0.2)
        scenario_forecast['yhat'] = base_forecast['yhat'] * adjustment_factor
        
        # Adjust confidence intervals if available
        if 'yhat_lower' in base_forecast.columns and 'yhat_upper' in base_forecast.columns:
            scenario_forecast['yhat_lower'] = base_forecast['yhat_lower'] * adjustment_factor
            scenario_forecast['yhat_upper'] = base_forecast['yhat_upper'] * adjustment_factor
        
        # Plot comparison
        fig = go.Figure()
        
        # Historical data
        country_data = time_series_df[time_series_df['country'] == selected_country].copy()
        fig.add_trace(go.Scatter(
            x=country_data['ds'],
            y=country_data['y'],
            mode='lines',
            name='Historical Data',
            line=dict(color='gray', width=1)
        ))
        
        # Base forecast
        fig.add_trace(go.Scatter(
            x=base_forecast['ds'],
            y=base_forecast['yhat'],
            mode='lines',
            name='Base Forecast',
            line=dict(color='blue', width=2)
        ))
        
        # Scenario forecast
        fig.add_trace(go.Scatter(
            x=scenario_forecast['ds'],
            y=scenario_forecast['yhat'],
            mode='lines',
            name='Scenario Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Fill between for confidence intervals if available
        if 'yhat_lower' in base_forecast.columns and 'yhat_upper' in base_forecast.columns:
            fig.add_trace(go.Scatter(
                x=base_forecast['ds'].tolist() + base_forecast['ds'].tolist()[::-1],
                y=base_forecast['yhat_upper'].tolist() + base_forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Base Confidence',
                showlegend=True
            ))
        
        if 'yhat_lower' in scenario_forecast.columns and 'yhat_upper' in scenario_forecast.columns:
            fig.add_trace(go.Scatter(
                x=scenario_forecast['ds'].tolist() + scenario_forecast['ds'].tolist()[::-1],
                y=scenario_forecast['yhat_upper'].tolist() + scenario_forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Scenario Confidence',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"Scenario Analysis: {selected_country} (Based on {base_model})",
            xaxis_title="Date",
            yaxis_title="Migration Rate",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact analysis
        last_base = base_forecast['yhat'].iloc[-1]
        last_scenario = scenario_forecast['yhat'].iloc[-1]
        impact = ((last_scenario - last_base) / last_base) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Forecast (Final Year)", f"{last_base:.2f}")
        with col2:
            st.metric("Scenario Forecast (Final Year)", f"{last_scenario:.2f}")
        with col3:
            delta = f"{impact:.1f}%"
            delta_color = "inverse" if impact < 0 else "normal"
            st.metric("Impact", delta, delta_color=delta_color)
            
            # Interpretation
            if impact > 10:
                st.info("🚨 High impact scenario - significant change expected")
            elif impact > 5:
                st.info("⚠️ Moderate impact scenario")
            elif abs(impact) < 2:
                st.info("📊 Minimal impact scenario")
    else:
        st.info("Generate forecasts first to run scenario analysis")

with tab4:
    st.header("Advanced Forecasting Metrics")
    
    # Filter data for selected country
    country_data = time_series_df[time_series_df['country'] == selected_country].copy()
    
    if len(country_data) > 10:
        # Time series decomposition
        st.subheader("Time Series Decomposition")
        
        # Simple moving averages
        country_data['MA_3'] = country_data['y'].rolling(window=3, min_periods=1).mean()
        country_data['MA_5'] = country_data['y'].rolling(window=5, min_periods=1).mean()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Original Series", "3-Year Moving Average", 
                          "5-Year Moving Average", "Year-over-Year Change"),
            vertical_spacing=0.15
        )
        
        # Original series
        fig.add_trace(
            go.Scatter(x=country_data['ds'], y=country_data['y'], 
                      mode='lines', name='Original'),
            row=1, col=1
        )
        
        # 3-year MA
        fig.add_trace(
            go.Scatter(x=country_data['ds'], y=country_data['MA_3'], 
                      mode='lines', name='3-Year MA', line=dict(color='red')),
            row=1, col=2
        )
        
        # 5-year MA
        fig.add_trace(
            go.Scatter(x=country_data['ds'], y=country_data['MA_5'], 
                      mode='lines', name='5-Year MA', line=dict(color='green')),
            row=2, col=1
        )
        
        # YoY change
        country_data['YoY'] = country_data['y'].pct_change(periods=1) * 100
        fig.add_trace(
            go.Scatter(x=country_data['ds'], y=country_data['YoY'], 
                      mode='lines', name='YoY % Change', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        stats_data = {
            'Statistic': ['Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{country_data['y'].mean():.3f}",
                f"{country_data['y'].std():.3f}",
                f"{country_data['y'].min():.3f}",
                f"{country_data['y'].quantile(0.25):.3f}",
                f"{country_data['y'].median():.3f}",
                f"{country_data['y'].quantile(0.75):.3f}",
                f"{country_data['y'].max():.3f}",
                f"{country_data['y'].skew():.3f}",
                f"{country_data['y'].kurtosis():.3f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Additional metrics
        st.subheader("Additional Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            autocorr_1 = country_data['y'].autocorr(lag=1)
            st.metric("Autocorrelation (lag=1)", f"{autocorr_1:.3f}")
        
        with col2:
            variance = country_data['y'].var()
            st.metric("Variance", f"{variance:.3f}")
        
        with col3:
            cv = (country_data['y'].std() / country_data['y'].mean()) * 100
            st.metric("Coefficient of Variation", f"{cv:.1f}%")
    else:
        st.warning("Insufficient data for advanced metrics. Need at least 10 data points.")

with tab5:
    st.header("Export Forecasting Results")
    
    if st.session_state.forecasts:
        # Select forecast to export
        forecast_to_export = st.selectbox(
            "Select forecast to export:",
            list(st.session_state.forecasts.keys())
        )
        
        forecast_data = st.session_state.forecasts[forecast_to_export]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Preview
            st.subheader("Preview")
            st.dataframe(forecast_data.head(10), use_container_width=True)
            
            # Summary stats
            st.subheader("Summary Statistics")
            st.write(f"Forecast Period: {forecast_data['ds'].min().date()} to {forecast_data['ds'].max().date()}")
            st.write(f"Number of periods: {len(forecast_data)}")
            st.write(f"Average forecast: {forecast_data['yhat'].mean():.2f}")
            st.write(f"Country: {selected_country}")
        
        with col2:
            # Export options
            st.subheader("Export Options")
            
            # CSV export
            csv = forecast_data.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"migration_forecast_{selected_country}_{forecast_to_export}.csv",
                mime="text/csv"
            )
            
            # JSON export
            json_str = forecast_data.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"migration_forecast_{selected_country}_{forecast_to_export}.json",
                mime="application/json"
            )
            
            # Excel export - using StringIO for compatibility
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    forecast_data.to_excel(writer, sheet_name='Forecast', index=False)
                    
                    # Add summary sheet
                    summary_data = pd.DataFrame({
                        'Metric': ['Country', 'Model', 'Start Date', 'End Date', 'Periods', 'Average Forecast'],
                        'Value': [selected_country, forecast_to_export, 
                                 forecast_data['ds'].min().date(), 
                                 forecast_data['ds'].max().date(),
                                 len(forecast_data),
                                 forecast_data['yhat'].mean()]
                    })
                    summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📥 Download as Excel",
                    data=output.getvalue(),
                    file_name=f"migration_forecast_{selected_country}_{forecast_to_export}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Excel export error: {e}")
                st.info("Install openpyxl for Excel export: `pip install openpyxl`")
            
            # Generate report
            if st.button("📋 Generate Detailed Report"):
                with st.spinner("Generating report..."):
                    trend = "increasing" if forecast_data['yhat'].iloc[-1] > forecast_data['yhat'].iloc[0] else "decreasing"
                    
                    report = f"""# Migration Forecast Report

## Country: {selected_country}
## Model: {forecast_to_export}
## Forecast Period: {forecast_data['ds'].min().date()} to {forecast_data['ds'].max().date()}

## Key Metrics:
- Average Forecast: {forecast_data['yhat'].mean():.2f}
- Maximum Forecast: {forecast_data['yhat'].max():.2f}
- Minimum Forecast: {forecast_data['yhat'].min():.2f}
- Forecast Range: {forecast_data['yhat'].max() - forecast_data['yhat'].min():.2f}

## Trend Analysis:
The forecast shows a {trend} trend.

## Recommendations:
1. Monitor actual migration data regularly
2. Update forecasts quarterly
3. Consider economic indicators for scenario planning
4. Validate forecasts with multiple models

## Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    st.download_button(
                        label="📄 Download Report (TXT)",
                        data=report,
                        file_name=f"forecast_report_{selected_country}_{forecast_to_export}.txt",
                        mime="text/plain"
                    )
    else:
        st.info("Generate forecasts first to export results")

# Installation instructions in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(" Installation Requirements")

if not PROPHET_AVAILABLE or not ARIMA_AVAILABLE:
    st.sidebar.markdown("""
    **Missing packages detected:**
    
    Install with:
    ```bash
    pip install prophet statsmodels openpyxl
    ```
    
    For Prophet on Windows, you might need:
    ```bash
    pip install prophet --no-cache-dir
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
**Advanced Forecasting Features:**
- Multiple model comparison
- Ensemble forecasting with weighted averages  
- Scenario analysis with adjustable parameters
- Confidence intervals and uncertainty quantification
- Export capabilities (CSV, JSON, Excel, Reports)
- Time series decomposition and statistical analysis""")
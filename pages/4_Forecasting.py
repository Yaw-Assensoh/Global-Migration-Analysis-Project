# pages/4_Forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.data_loader import load_data

# Page configuration
st.set_page_config(page_title="Forecasting", page_icon="", layout="wide")

# Title
st.markdown('<h1 class="main-header"> Migration Forecasting & Scenarios</h1>', unsafe_allow_html=True)

# Load data
df, summary = load_data()

if df is not None:
    # Sidebar
    st.sidebar.markdown("###  Forecasting Settings")
    
    forecast_type = st.sidebar.selectbox(
        "Forecast Model:",
        ["Linear Projection", "Growth Rate Model", "Scenario Analysis", "Migration Impact"]
    )
    
    years_to_forecast = st.sidebar.slider(
        "Years to Forecast:",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )
    
    # Main content
    if forecast_type == "Linear Projection":
        st.markdown("###  Linear Projection Model")
        
        # Select countries
        countries = st.multiselect(
            "Select countries to forecast:",
            df['Country'].unique().tolist(),
            default=['United States', 'India', 'Germany', 'United Arab Emirates']
        )
        
        if countries:
            forecast_df = df[df['Country'].isin(countries)].copy()
            
            # Create forecast
            years = list(range(2025, 2025 + years_to_forecast + 1))
            
            fig = go.Figure()
            
            for _, row in forecast_df.iterrows():
                # Simple linear forecast based on current migration
                current_migration = row['Net_Migrants']
                forecast_values = [current_migration * (1 + 0.02 * i) for i in range(len(years))]
                
                fig.add_trace(go.Scatter(
                    x=years,
                    y=forecast_values,
                    mode='lines+markers',
                    name=row['Country'],
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title=f'Migration Forecast ({years_to_forecast} years)',
                xaxis_title="Year",
                yaxis_title="Projected Net Migration",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.markdown("#### Forecast Summary")
            
            summary_data = []
            for _, row in forecast_df.iterrows():
                current = row['Net_Migrants']
                projected = current * (1 + 0.02 * years_to_forecast)
                change = ((projected - current) / current) * 100
                
                summary_data.append({
                    'Country': row['Country'],
                    'Current Migration': f"{current/1e6:.1f}M",
                    'Projected Migration': f"{projected/1e6:.1f}M",
                    'Change (%)': f"{change:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    elif forecast_type == "Growth Rate Model":
        st.markdown("### Growth Rate Based Forecast")
        
        # Calculate growth rates
        df['Growth_Rate'] = df['Yearly Change'] / 100
        
        # Create forecast scenarios
        scenario = st.selectbox(
            "Select Scenario:",
            ["Current Trends", "Increased Migration", "Decreased Migration", "Stabilization"]
        )
        
        # Adjust growth rates based on scenario
        if scenario == "Increased Migration":
            multiplier = 1.5
        elif scenario == "Decreased Migration":
            multiplier = 0.5
        elif scenario == "Stabilization":
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        # Show top/bottom forecast changes
        df['Projected_Change'] = df['Net_Migrants'] * (1 + (df['Growth_Rate'] * multiplier * years_to_forecast))
        df['Change_Percentage'] = ((df['Projected_Change'] - df['Net_Migrants']) / df['Net_Migrants']) * 100
        
        # Top gainers and losers
        col1, col2 = st.columns(2)
        
        with col1:
            top_gainers = df.nlargest(10, 'Change_Percentage')
            st.markdown("####  Top Projected Gainers")
            
            fig = px.bar(
                top_gainers,
                x='Change_Percentage',
                y='Country',
                orientation='h',
                color='Change_Percentage',
                color_continuous_scale='Greens',
                title=f'Top Gainers ({scenario} Scenario)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_losers = df.nsmallest(10, 'Change_Percentage')
            st.markdown("####  Top Projected Losers")
            
            fig = px.bar(
                top_losers,
                x='Change_Percentage',
                y='Country',
                orientation='h',
                color='Change_Percentage',
                color_continuous_scale='Reds',
                title=f'Top Losers ({scenario} Scenario)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif forecast_type == "Scenario Analysis":
        st.markdown("###  Scenario Analysis")
        
        # Define scenarios
        scenarios = {
            "Optimistic": {"migration_growth": 0.03, "urban_growth": 0.02},
            "Pessimistic": {"migration_growth": -0.01, "urban_growth": 0.005},
            "Neutral": {"migration_growth": 0.01, "urban_growth": 0.01},
            "High Urbanization": {"migration_growth": 0.02, "urban_growth": 0.03}
        }
        
        selected_scenario = st.selectbox("Select Scenario:", list(scenarios.keys()))
        
        # Calculate impacts
        params = scenarios[selected_scenario]
        
        # Create impact visualization
        impact_data = []
        for _, row in df.iterrows():
            current_migration = row['Net_Migrants']
            current_urban = row['Urban_Pop_Percent']
            
            projected_migration = current_migration * (1 + params['migration_growth'] * years_to_forecast)
            projected_urban = min(100, current_urban * (1 + params['urban_growth'] * years_to_forecast))
            
            impact_data.append({
                'Country': row['Country'],
                'Current_Migration': current_migration,
                'Projected_Migration': projected_migration,
                'Current_Urban': current_urban,
                'Projected_Urban': projected_urban,
                'Migration_Change': projected_migration - current_migration,
                'Urban_Change': projected_urban - current_urban
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Show scenario impacts
        st.markdown(f"#### {selected_scenario} Scenario Impacts")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_migration_change = impact_df['Migration_Change'].sum()
            st.metric("Total Migration Change", f"{total_migration_change/1e6:.1f}M")
        
        with col2:
            avg_urban_change = impact_df['Urban_Change'].mean()
            st.metric("Avg Urbanization Change", f"{avg_urban_change:.1f}%")
        
        with col3:
            affected_countries = len(impact_df[abs(impact_df['Migration_Change']) > 10000])
            st.metric("Significantly Affected Countries", affected_countries)
        
        # Show impacts by continent
        if 'Continent' in df.columns:
            continent_impacts = impact_df.merge(df[['Country', 'Continent']], on='Country')
            continent_summary = continent_impacts.groupby('Continent').agg({
                'Migration_Change': 'sum',
                'Urban_Change': 'mean'
            }).round(2)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Total Migration Change by Continent', 'Avg Urbanization Change by Continent')
            )
            
            fig.add_trace(
                go.Bar(x=continent_summary.index, y=continent_summary['Migration_Change']/1e6,
                      name='Migration Change'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=continent_summary.index, y=continent_summary['Urban_Change'],
                      name='Urban Change'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Migration Impact
        st.markdown("### Migration Impact Analysis")
        
        # Calculate migration impacts
        df['Migration_Impact'] = (df['Net_Migrants'] / df['Population']) * 100
        
        # Categorize impact levels
        df['Impact_Level'] = pd.cut(
            df['Migration_Impact'],
            bins=[-float('inf'), -1, -0.1, 0.1, 1, float('inf')],
            labels=['Very High Negative', 'Moderate Negative', 'Neutral', 'Moderate Positive', 'Very High Positive']
        )
        
        # Impact distribution
        impact_counts = df['Impact_Level'].value_counts().sort_index()
        
        fig = px.bar(
            x=impact_counts.index,
            y=impact_counts.values,
            color=impact_counts.index,
            color_discrete_sequence=['#A23B72', '#D4A5A5', '#CBD5E1', '#A5B4FC', '#2E86AB'],
            title='Migration Impact Distribution',
            labels={'x': 'Impact Level', 'y': 'Number of Countries'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact by population
        st.markdown("####  Impact vs Population Size")
        
        fig = px.scatter(
            df,
            x='Population',
            y='Migration_Impact',
            size=abs(df['Migration_Impact']),
            color='Continent',
            hover_name='Country',
            title='Migration Impact Relative to Population Size',
            labels={'Population': 'Total Population', 'Migration_Impact': 'Migration Impact (%)'},
            log_x=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.markdown("####  Impact Insights")
        
        high_positive = df[df['Impact_Level'] == 'Very High Positive']
        high_negative = df[df['Impact_Level'] == 'Very High Negative']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(high_positive) > 0:
                st.write("**Countries with Very High Positive Impact:**")
                for _, row in high_positive.head(5).iterrows():
                    st.write(f"• {row['Country']} ({row['Migration_Impact']:.2f}%)")
        
        with col2:
            if len(high_negative) > 0:
                st.write("**Countries with Very High Negative Impact:**")
                for _, row in high_negative.head(5).iterrows():
                    st.write(f"• {row['Country']} ({row['Migration_Impact']:.2f}%)")

else:
    st.error("Data not loaded. Please run the Jupyter notebook first.")
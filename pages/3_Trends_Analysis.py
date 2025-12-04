# pages/3_Trends_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data
from utils.visualizations import create_correlation_heatmap

# Page configuration
st.set_page_config(page_title="Trends Analysis", page_icon="", layout="wide")

# Title
st.markdown('<h1 class="main-header"> Demographic Trends & Correlations</h1>', unsafe_allow_html=True)

# Load data
df, summary = load_data()

if df is not None:
    # Sidebar
    st.sidebar.markdown("###  Analysis Settings")
    
    analysis_type = st.sidebar.selectbox(
        "Analysis Type:",
        ["Correlation Analysis", "Scatter Analysis", "Distribution Analysis", "Comparative Analysis"]
    )
    
    # Main content
    if analysis_type == "Correlation Analysis":
        st.markdown("###  Correlation Matrix")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = create_correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("####  Correlation Guide")
            st.markdown("""
            **Strength Guidelines:**
            - ±0.7 to ±1.0: Strong
            - ±0.3 to ±0.7: Moderate
            - ±0.0 to ±0.3: Weak
            
            **Key Correlations to Watch:**
            1. **Migration vs Urbanization**
            2. **Migration vs Median Age**
            3. **Migration vs Fertility Rate**
            4. **Urbanization vs Median Age**
            """)
    
    elif analysis_type == "Scatter Analysis":
        st.markdown("###  Scatter Plot Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox(
                "X-axis Variable:",
                ['Population', 'Density', 'Urban_Pop_Percent', 'Median_Age', 'Fertility_Rate'],
                index=2
            )
        
        with col2:
            y_var = st.selectbox(
                "Y-axis Variable:",
                ['Net_Migrants', 'Migration_Rate_per_1000'],
                index=1
            )
        
        with col3:
            color_var = st.selectbox(
                "Color by:",
                ['Continent', 'Net_Migrants', 'Migration_Rate_per_1000', 'None'],
                index=0
            )
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=x_var,
            y=y_var,
            size='Population',
            color=color_var if color_var != 'None' else None,
            hover_name='Country',
            trendline='ols',
            title=f'{y_var} vs {x_var}',
            labels={
                x_var: x_var.replace('_', ' ').title(),
                y_var: y_var.replace('_', ' ').title()
            }
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        correlation = df[x_var].corr(df[y_var])
        st.metric(f"Correlation ({x_var} vs {y_var})", f"{correlation:.3f}")
    
    elif analysis_type == "Distribution Analysis":
        st.markdown("###  Distribution Analysis")
        
        variable = st.selectbox(
            "Select Variable to Analyze:",
            ['Net_Migrants', 'Migration_Rate_per_1000', 'Population', 'Density', 
             'Urban_Pop_Percent', 'Median_Age', 'Fertility_Rate'],
            index=1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                df,
                x=variable,
                nbins=30,
                title=f'Distribution of {variable.replace("_", " ").title()}',
                labels={variable: variable.replace('_', ' ').title()}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot by continent
            fig = px.box(
                df,
                x='Continent',
                y=variable,
                color='Continent',
                title=f'{variable.replace("_", " ").title()} by Continent',
                labels={variable: variable.replace('_', ' ').title()}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("####  Statistical Summary")
        
        stats = df[variable].describe()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
        with col2:
            st.metric("Median", f"{stats['50%']:.2f}")
        with col3:
            st.metric("Std Dev", f"{stats['std']:.2f}")
        with col4:
            st.metric("Range", f"{stats['max'] - stats['min']:.2f}")
    
    else:  # Comparative Analysis
        st.markdown("###  Comparative Analysis")
        
        # Select countries to compare
        default_countries = ['United States', 'India', 'Germany', 'United Arab Emirates', 'Nigeria']
        selected_countries = st.multiselect(
            "Select countries to compare:",
            df['Country'].unique().tolist(),
            default=default_countries
        )
        
        if selected_countries:
            compare_df = df[df['Country'].isin(selected_countries)].copy()
            
            # Radar chart for multiple metrics
            metrics = ['Net_Migrants', 'Migration_Rate_per_1000', 'Urban_Pop_Percent', 
                      'Median_Age', 'Fertility_Rate', 'Density']
            
            # Normalize metrics for radar chart
            normalized_data = []
            for metric in metrics:
                if metric in compare_df.columns:
                    max_val = compare_df[metric].max()
                    min_val = compare_df[metric].min()
                    if max_val != min_val:
                        normalized = (compare_df[metric] - min_val) / (max_val - min_val)
                    else:
                        normalized = compare_df[metric] * 0
                    normalized_data.append(normalized)
            
            # Create radar chart
            fig = go.Figure()
            
            for i, country in enumerate(selected_countries):
                country_data = [data.iloc[i] for data in normalized_data]
                fig.add_trace(go.Scatterpolar(
                    r=country_data + [country_data[0]],  # Close the circle
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=country
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title="Country Comparison (Normalized Metrics)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.markdown("####  Comparison Data")
            
            display_cols = ['Country', 'Net_Migrants', 'Migration_Rate_per_1000', 
                          'Population', 'Urban_Pop_Percent', 'Median_Age', 'Fertility_Rate']
            
            display_df = compare_df[display_cols].copy()
            display_df['Net_Migrants'] = display_df['Net_Migrants'].apply(lambda x: f'{x/1e6:.1f}M')
            display_df['Population'] = display_df['Population'].apply(lambda x: f'{x/1e6:.1f}M')
            display_df['Migration_Rate_per_1000'] = display_df['Migration_Rate_per_1000'].round(1)
            display_df['Urban_Pop_Percent'] = display_df['Urban_Pop_Percent'].round(1)
            display_df['Median_Age'] = display_df['Median_Age'].round(1)
            display_df['Fertility_Rate'] = display_df['Fertility_Rate'].round(2)
            
            st.dataframe(display_df, use_container_width=True)

else:
    st.error("Data not loaded. Please run the Jupyter notebook first.")
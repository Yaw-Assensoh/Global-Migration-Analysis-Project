# pages/2__Migration_Flows.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_data
from utils.helpers import create_sidebar_filters

# Page configuration
st.set_page_config(page_title="Migration Flows", page_icon="", layout="wide")

# Title
st.markdown('<h1 class="main-header">Migration Flows Analysis</h1>', unsafe_allow_html=True)

# Load data
df, summary = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.markdown("###  Analysis Settings")
    
    # Flow type selection
    flow_type = st.sidebar.radio(
        "Analysis Focus:",
        ["Immigration Flows", "Emigration Flows", "Net Migration"],
        index=0
    )
    
    # Additional filters
    min_migrants = st.sidebar.number_input(
        "Minimum migrants (absolute):",
        min_value=0,
        value=100000,
        step=50000
    )
    
    # Region focus
    focus_region = st.sidebar.selectbox(
        "Focus Region:",
        ["Global", "Asia", "Europe", "North America", "Africa", "South America", "Oceania"]
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Flow Magnitude", "Flow Intensity", "Flow Patterns"])
    
    with tab1:
        st.markdown("###  Migration Flow Magnitude")
        
        # Prepare data based on flow type
        if flow_type == "Immigration Flows":
            flow_df = df[df['Net_Migrants'] > 0].copy()
            flow_df = flow_df[flow_df['Net_Migrants'] >= min_migrants]
            color_title = "Immigration"
            color_scale = "Blues"
        elif flow_type == "Emigration Flows":
            flow_df = df[df['Net_Migrants'] < 0].copy()
            flow_df['Net_Migrants_Abs'] = abs(flow_df['Net_Migrants'])
            flow_df = flow_df[flow_df['Net_Migrants_Abs'] >= min_migrants]
            color_title = "Emigration"
            color_scale = "Reds"
        else:
            flow_df = df.copy()
            flow_df = flow_df[abs(flow_df['Net_Migrants']) >= min_migrants]
            color_title = "Net Migration"
            color_scale = "RdBu"
        
        if focus_region != "Global":
            flow_df = flow_df[flow_df['Continent'] == focus_region]
        
        # Create bubble chart
        fig = px.scatter(
            flow_df,
            x='Population',
            y='Net_Migrants',
            size=abs(flow_df['Net_Migrants']),
            color='Net_Migrants' if flow_type == "Net Migration" else abs(flow_df['Net_Migrants']),
            hover_name='Country',
            color_continuous_scale=color_scale,
            labels={
                'Population': 'Total Population',
                'Net_Migrants': 'Net Migration'
            },
            title=f'{flow_type} - {focus_region}',
            log_x=True
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Countries Shown", len(flow_df))
        with col2:
            st.metric("Total Flow", f"{flow_df['Net_Migrants'].sum()/1e6:.1f}M")
        with col3:
            st.metric("Avg per Country", f"{flow_df['Net_Migrants'].mean()/1e6:.1f}M")
    
    with tab2:
        st.markdown("###  Migration Flow Intensity")
        
        # Calculate migration rates
        intensity_df = df.copy()
        intensity_df['Migration_Rate'] = intensity_df['Migration_Rate_per_1000']
        
        if focus_region != "Global":
            intensity_df = intensity_df[intensity_df['Continent'] == focus_region]
        
        # Sort by intensity
        if flow_type == "Immigration Flows":
            intensity_df = intensity_df[intensity_df['Migration_Rate'] > 0]
            intensity_df = intensity_df.sort_values('Migration_Rate', ascending=False)
            title_suffix = "Immigration Intensity"
        elif flow_type == "Emigration Flows":
            intensity_df = intensity_df[intensity_df['Migration_Rate'] < 0]
            intensity_df = intensity_df.sort_values('Migration_Rate', ascending=True)
            title_suffix = "Emigration Intensity"
        else:
            intensity_df = intensity_df.sort_values('Migration_Rate', key=abs, ascending=False)
            title_suffix = "Migration Intensity"
        
        # Show top 20
        top_intensity = intensity_df.head(20)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_intensity['Country'],
            x=top_intensity['Migration_Rate'],
            orientation='h',
            marker_color=['#2E86AB' if x > 0 else '#A23B72' for x in top_intensity['Migration_Rate']],
            text=[f'{x:.1f}' for x in top_intensity['Migration_Rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Top 20 Countries by {title_suffix}',
            xaxis_title="Migration Rate (per 1000 population)",
            yaxis_title="Country",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("###  Migration Flow Patterns")
        
        # Create a matrix-style visualization
        patterns_df = df.copy()
        
        if focus_region != "Global":
            patterns_df = patterns_df[patterns_df['Continent'] == focus_region]
        
        # Categorize countries
        patterns_df['Migration_Type'] = pd.cut(
            patterns_df['Net_Migrants'],
            bins=[-float('inf'), -100000, 0, 100000, float('inf')],
            labels=['Major Emigration', 'Moderate Emigration', 'Moderate Immigration', 'Major Immigration']
        )
        
        # Count by type
        type_counts = patterns_df['Migration_Type'].value_counts().sort_index()
        
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            color=type_counts.index,
            color_discrete_sequence=['#A23B72', '#D4A5A5', '#A5B4FC', '#2E86AB'],
            text=type_counts.values,
            title=f'Migration Pattern Distribution - {focus_region}'
        )
        
        fig.update_layout(
            xaxis_title="Migration Type",
            yaxis_title="Number of Countries",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern insights
        st.markdown("####  Pattern Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            major_imm = patterns_df[patterns_df['Migration_Type'] == 'Major Immigration']
            if len(major_imm) > 0:
                st.write("**Major Immigration Countries:**")
                for country in major_imm['Country'].head(5):
                    st.write(f"• {country}")
        
        with col2:
            major_emm = patterns_df[patterns_df['Migration_Type'] == 'Major Emigration']
            if len(major_emm) > 0:
                st.write("**Major Emigration Countries:**")
                for country in major_emm['Country'].head(5):
                    st.write(f"• {country}")

else:
    st.error("Data not loaded. Please run the Jupyter notebook first.")
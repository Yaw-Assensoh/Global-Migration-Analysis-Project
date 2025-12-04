# pages/1_Overview.py
import streamlit as st
import pandas as pd
from utils.data_loader import load_data, get_filtered_data
from utils.visualizations import create_migration_leaders_chart, create_continent_analysis
from utils.helpers import create_sidebar_filters, format_number

# Page configuration
st.set_page_config(page_title="Overview Dashboard", page_icon="", layout="wide")

# Title
st.markdown('<h1 class="main-header"> Dashboard Overview</h1>', unsafe_allow_html=True)

# Load data
df, summary = load_data()

if df is not None and summary is not None:
    # Sidebar filters
    st.sidebar.markdown("###  Filters")
    filters = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = get_filtered_data(df, filters)
    
    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Filtered Stats")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Countries", len(filtered_df))
    with col2:
        st.metric("Avg Migration", f"{filtered_df['Net_Migrants'].mean()/1e6:.1f}M")
    
    # Main content - Top metrics
    st.markdown("### Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_migration = filtered_df['Net_Migrants'].sum()
        st.metric("Total Net Migration", f"{format_number(total_migration)}", 
                 delta=f"{total_migration/1e6:.1f}M")
    
    with col2:
        avg_rate = filtered_df['Migration_Rate_per_1000'].mean()
        st.metric("Avg Migration Rate", f"{avg_rate:.1f}/1000", 
                 delta_color="normal" if avg_rate > 0 else "inverse")
    
    with col3:
        immigration_countries = len(filtered_df[filtered_df['Net_Migrants'] > 0])
        st.metric("Immigration Countries", immigration_countries)
    
    with col4:
        emigration_countries = len(filtered_df[filtered_df['Net_Migrants'] < 0])
        st.metric("Emigration Countries", emigration_countries)
    
    # Migration Leaders
    st.markdown("### 🏆 Migration Leaders")
    fig_leaders = create_migration_leaders_chart(filtered_df, filters['top_n'])
    st.plotly_chart(fig_leaders, use_container_width=True)
    
    # Continent Analysis
    st.markdown("### Continental Analysis")
    fig_continent = create_continent_analysis(filtered_df)
    st.plotly_chart(fig_continent, use_container_width=True)
    
    # Quick Insights
    st.markdown("###  Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ####  **Top Patterns**
        - **Highest Immigration:** United States (1.2M)
        - **Highest Emigration:** India (495K)
        - **Urban Attraction:** 67.5% avg urbanization in migrant destinations
        - **Age Factor:** 38.5 years avg median age in immigration countries
        """)
    
    with col2:
        # Calculate some insights dynamically
        top_imm = filtered_df.nlargest(1, 'Net_Migrants').iloc[0]
        top_emm = filtered_df.nsmallest(1, 'Net_Migrants').iloc[0]
        
        st.markdown(f"""
        ####  **Current Filter Insights**
        - **Top in selection:** {top_imm['Country']} ({top_imm['Net_Migrants']/1e6:.1f}M)
        - **Bottom in selection:** {top_emm['Country']} ({abs(top_emm['Net_Migrants'])/1e6:.1f}M)
        - **Avg Density:** {filtered_df['Density'].mean():.0f} people/km²
        - **Avg Fertility:** {filtered_df['Fertility_Rate'].mean():.1f} children/woman
        """)
    
else:
    st.error("Data not loaded. Please run the Jupyter notebook first.")
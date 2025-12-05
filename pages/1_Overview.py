# pages/1_Overview.py - 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from utils
try:
    from utils.data_loader import load_data, get_filtered_data
except ImportError:
    st.error(" Could not import from utils. Make sure the utils folder exists.")
    # Define fallback functions
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv('data/processed/cleaned_migration_data.csv')
            summary = {
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'total_countries': len(df),
                'global_population': df['Population'].sum() / 1e9
            }
            return df, summary
        except:
            st.error("Data not found. Please run the notebook first.")
            return None, None
    
    def get_filtered_data(df, **kwargs):
        return df

# Page configuration
st.set_page_config(
    page_title="Overview Dashboard",
    page_icon="",
    layout="wide"
)

# Title
st.markdown('#  Dashboard Overview')

# Load data
with st.spinner('Loading data...'):
    df, summary = load_data()

if df is not None and summary is not None:
    # Sidebar filters
    st.sidebar.markdown("##  Filters")
    
    # Continent filter
    if 'Continent' in df.columns:
        continents = ['All'] + sorted(df['Continent'].dropna().unique().tolist())
        selected_continent = st.sidebar.selectbox("Select Continent", continents, index=0)
    else:
        selected_continent = 'All'
        st.sidebar.info("Continent data not available")
    
    # Migration status filter
    migration_status_options = ['All', 'Immigration Only', 'Emigration Only', 'Balanced']
    selected_status = st.sidebar.selectbox("Migration Status", migration_status_options, index=0)
    
    # Population range filter
    if 'Population' in df.columns:
        min_pop = int(df['Population'].min() / 1e6)
        max_pop = int(df['Population'].max() / 1e6)
        population_range = st.sidebar.slider(
            "Population Range (millions)",
            min_value=min_pop,
            max_value=max_pop,
            value=(10, max_pop)
        )
    else:
        population_range = (10, 1000)
    
    # Top N countries filter
    top_n = st.sidebar.slider("Show Top N Countries", 5, 20, 10)
    
    # Apply filters
    filtered_df = get_filtered_data(
        df, 
        continent=selected_continent,
        migration_status=selected_status,
        population_range=population_range
    )
    
    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("##  Filtered Stats")
    st.sidebar.metric("Countries", len(filtered_df))
    
    if 'Net_Migrants' in filtered_df.columns:
        avg_migration = filtered_df['Net_Migrants'].mean() / 1e6
        st.sidebar.metric("Avg Migration", f"{avg_migration:.1f}M")
    
    # Main content - Top metrics
    st.markdown("##  Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Population' in filtered_df.columns:
            total_pop = filtered_df['Population'].sum() / 1e9
            st.metric("Total Population", f"{total_pop:.1f}B")
    
    with col2:
        if 'Net_Migrants' in filtered_df.columns:
            total_migration = filtered_df['Net_Migrants'].sum()
            st.metric("Total Net Migration", f"{total_migration/1e6:.1f}M")
    
    with col3:
        immigration_countries = len(filtered_df[filtered_df['Net_Migrants'] > 0]) if 'Net_Migrants' in filtered_df.columns else 0
        st.metric("Immigration Countries", immigration_countries)
    
    with col4:
        emigration_countries = len(filtered_df[filtered_df['Net_Migrants'] < 0]) if 'Net_Migrants' in filtered_df.columns else 0
        st.metric("Emigration Countries", emigration_countries)
    
    # Migration Leaders Visualization
    st.markdown("##  Migration Leaders")
    
    if 'Net_Migrants' in filtered_df.columns and 'Country' in filtered_df.columns:
        # Get top immigration and emigration countries
        top_imm = filtered_df.nlargest(top_n, 'Net_Migrants')
        top_emm = filtered_df.nsmallest(top_n, 'Net_Migrants')
        
        # Create the visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Top {top_n} Immigration Countries', f'Top {top_n} Emigration Countries')
        )
        
        # Immigration bars
        fig.add_trace(
            go.Bar(
                y=top_imm['Country'],
                x=top_imm['Net_Migrants'] / 1e6,
                orientation='h',
                marker_color='#2E86AB',
                text=[f'{x/1e6:.1f}M' for x in top_imm['Net_Migrants']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Emigration bars
        fig.add_trace(
            go.Bar(
                y=top_emm['Country'],
                x=abs(top_emm['Net_Migrants']) / 1e6,
                orientation='h',
                marker_color='#A23B72',
                text=[f'{abs(x)/1e6:.1f}M' for x in top_emm['Net_Migrants']],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            xaxis_title="Net Migrants (Millions)",
            yaxis_title="Country"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Migration data not available for visualization")
    
    # Data Table
    st.markdown("##  Data Preview")
    
    # Select columns to show
    available_columns = filtered_df.columns.tolist()
    default_columns = ['Country', 'Population', 'Net_Migrants', 'Migration_Rate_per_1000']
    
    # Only include columns that exist
    display_columns = [col for col in default_columns if col in available_columns]
    
    if display_columns:
        st.dataframe(
            filtered_df[display_columns].head(20),
            use_container_width=True
        )
    else:
        st.dataframe(filtered_df.head(20), use_container_width=True)
    
    # Quick Insights
    st.markdown("##  Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  **Migration Patterns**
        
        - **Top Destination:** United States
        - **Top Source:** India
        - **Economic Flow:** Developing → Developed countries
        - **Urban Focus:** Migrants prefer urban areas
        """)
    
    with col2:
        if 'Net_Migrants' in filtered_df.columns:
            avg_migration_rate = filtered_df['Migration_Rate_per_1000'].mean() if 'Migration_Rate_per_1000' in filtered_df.columns else 0
            st.markdown(f"""
            ###  **Current Selection**
            
            - **Countries:** {len(filtered_df)}
            - **Avg Migration Rate:** {avg_migration_rate:.1f}/1000
            - **Immigration Countries:** {immigration_countries}
            - **Emigration Countries:** {emigration_countries}
            """)
    
else:
    st.error("""
    ##  Data Not Loaded
    
    Please ensure:
    1. You have run the Jupyter notebook first
    2. The file `data/processed/cleaned_migration_data.csv` exists
    3. You have the required permissions to read the file
    
    **Steps to fix:**
    ```bash
    cd notebooks
    jupyter notebook exploratory_analysis.ipynb
    # Run all cells
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B;">
<small>Global Migration Analytics Platform • Data Source: World Population Review 2025</small>
</div>
""", unsafe_allow_html=True)
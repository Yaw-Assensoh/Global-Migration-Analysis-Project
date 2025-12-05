# utils/data_loader.py
import pandas as pd
import json
import streamlit as st

@st.cache_data
def load_data():
    """Load processed data with caching"""
    try:
        df = pd.read_csv('data/processed/cleaned_migration_data.csv')
        
        # Create a simple summary if file doesn't exist
        try:
            with open('data/processed/dashboard_summary.json', 'r') as f:
                summary = json.load(f)
        except:
            # Create basic summary from data
            summary = {
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'total_countries': len(df),
                'global_population': df['Population'].sum() / 1e9,
                'top_immigration': {
                    'country': df.nlargest(1, 'Net_Migrants').iloc[0]['Country'],
                    'value': float(df.nlargest(1, 'Net_Migrants').iloc[0]['Net_Migrants'])
                },
                'top_emigration': {
                    'country': df.nsmallest(1, 'Net_Migrants').iloc[0]['Country'],
                    'value': float(df.nsmallest(1, 'Net_Migrants').iloc[0]['Net_Migrants'])
                }
            }
        
        return df, summary
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.info("""
         **Please run the Jupyter notebook first:**
        1. Go to `notebooks/exploratory_analysis.ipynb`
        2. Run all cells to generate cleaned data
        3. Check that `data/processed/cleaned_migration_data.csv` exists
        """)
        return None, None
    except Exception as e:
        st.error(f" Error loading data: {e}")
        return None, None

def get_filtered_data(df, continent='All', migration_status='All', population_range=(10, 1000)):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    if continent != 'All' and 'Continent' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Continent'] == continent]
    
    if migration_status == 'Immigration Only':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] > 0]
    elif migration_status == 'Emigration Only':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] < 0]
    elif migration_status == 'Balanced':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] == 0]
    
    min_pop, max_pop = population_range
    filtered_df = filtered_df[
        (filtered_df['Population'] >= min_pop * 1e6) &
        (filtered_df['Population'] <= max_pop * 1e6)
    ]
    
    return filtered_df
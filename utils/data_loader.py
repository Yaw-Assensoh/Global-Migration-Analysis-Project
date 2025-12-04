# utils/data_loader.py
import pandas as pd
import json
import streamlit as st

@st.cache_data
def load_data():
    """Load processed data with caching"""
    try:
        df = pd.read_csv('data/processed/cleaned_migration_data.csv')
        with open('data/processed/dashboard_summary.json', 'r') as f:
            summary = json.load(f)
        return df, summary
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None

def get_filtered_data(df, filters):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    if filters.get('continent') and filters['continent'] != 'All':
        filtered_df = filtered_df[filtered_df['Continent'] == filters['continent']]
    
    if filters.get('migration_status') == 'Immigration Only':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] > 0]
    elif filters.get('migration_status') == 'Emigration Only':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] < 0]
    elif filters.get('migration_status') == 'Balanced':
        filtered_df = filtered_df[filtered_df['Net_Migrants'] == 0]
    
    if filters.get('population_range'):
        min_pop, max_pop = filters['population_range']
        filtered_df = filtered_df[
            (filtered_df['Population'] >= min_pop * 1e6) &
            (filtered_df['Population'] <= max_pop * 1e6)
        ]
    
    return filtered_df
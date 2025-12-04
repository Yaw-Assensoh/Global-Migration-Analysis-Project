# utils/helpers.py
import pandas as pd
import numpy as np
from datetime import datetime

def create_sidebar_filters(df):
    """Create standard sidebar filters"""
    filters = {}
    
    # Continent filter
    continents = ['All'] + sorted(df['Continent'].dropna().unique().tolist())
    selected_continent = st.sidebar.selectbox("Select Continent", continents, index=0)
    filters['continent'] = selected_continent
    
    # Migration status filter
    migration_status_options = ['All', 'Immigration Only', 'Emigration Only', 'Balanced']
    selected_status = st.sidebar.selectbox("Migration Status", migration_status_options, index=0)
    filters['migration_status'] = selected_status
    
    # Population range filter
    min_pop = int(df['Population'].min() / 1e6)
    max_pop = int(df['Population'].max() / 1e6)
    population_range = st.sidebar.slider(
        "Population Range (millions)",
        min_value=min_pop,
        max_value=max_pop,
        value=(10, max_pop)
    )
    filters['population_range'] = population_range
    
    # Top N countries filter
    top_n = st.sidebar.slider("Show Top N Countries", 5, 20, 10)
    filters['top_n'] = top_n
    
    return filters

def format_number(num):
    """Format large numbers for display"""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.0f}"
# pages/5_Data_Explorer.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_loader import load_data

# Page configuration
st.set_page_config(page_title="Data Explorer", page_icon="", layout="wide")

# Title
st.markdown('<h1 class="main-header"> Interactive Data Explorer</h1>', unsafe_allow_html=True)

# Load data
df, summary = load_data()

if df is not None:
    # Sidebar - Data controls
    st.sidebar.markdown("###  Data Controls")
    
    # Column selection
    all_columns = df.columns.tolist()
    default_columns = ['Country', 'Continent', 'Population', 'Net_Migrants', 
                      'Migration_Rate_per_1000', 'Urban_Pop_Percent', 'Median_Age']
    
    selected_columns = st.sidebar.multiselect(
        "Select columns to display:",
        all_columns,
        default=default_columns
    )
    
    # Row limit
    row_limit = st.sidebar.slider(
        "Maximum rows to display:",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    # Sort options
    sort_column = st.sidebar.selectbox(
        "Sort by:",
        selected_columns if selected_columns else all_columns,
        index=0 if selected_columns else 2
    )
    
    sort_order = st.sidebar.radio(
        "Sort order:",
        ["Ascending", "Descending"],
        horizontal=True
    )
    
    # Filter options
    st.sidebar.markdown("###  Data Filters")
    
    # Continent filter
    continents = ['All'] + sorted(df['Continent'].dropna().unique().tolist())
    selected_continent = st.sidebar.selectbox("Continent:", continents, index=0)
    
    # Migration range filter
    if 'Net_Migrants' in df.columns:
        min_mig = int(df['Net_Migrants'].min())
        max_mig = int(df['Net_Migrants'].max())
        migration_range = st.sidebar.slider(
            "Net Migration Range:",
            min_value=min_mig,
            max_value=max_mig,
            value=(min_mig, max_mig)
        )
    
    # Population filter
    if 'Population' in df.columns:
        min_pop = int(df['Population'].min())
        max_pop = int(df['Population'].max())
        population_filter = st.sidebar.checkbox("Filter by Population")
        
        if population_filter:
            population_range = st.sidebar.slider(
                "Population Range:",
                min_value=min_pop,
                max_value=max_pop,
                value=(int(min_pop), int(max_pop * 0.1))  # Default to lower 10%
            )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([" Data Table", " Summary Stats", " Data Quality", " Export Data"])
    
    with tab1:
        st.markdown("###  Interactive Data Table")
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_continent != 'All' and 'Continent' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Continent'] == selected_continent]
        
        if 'Net_Migrants' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['Net_Migrants'] >= migration_range[0]) &
                (filtered_df['Net_Migrants'] <= migration_range[1])
            ]
        
        if population_filter and 'Population' in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df['Population'] >= population_range[0]) &
                (filtered_df['Population'] <= population_range[1])
            ]
        
        # Sort data
        ascending = (sort_order == "Ascending")
        filtered_df = filtered_df.sort_values(sort_column, ascending=ascending)
        
        # Limit rows
        filtered_df = filtered_df.head(row_limit)
        
        # Select columns
        if selected_columns:
            display_df = filtered_df[selected_columns].copy()
        else:
            display_df = filtered_df.copy()
        
        # Format numbers
        def format_numbers(val):
            if isinstance(val, (int, float)):
                if abs(val) >= 1e9:
                    return f"{val/1e9:.1f}B"
                elif abs(val) >= 1e6:
                    return f"{val/1e6:.1f}M"
                elif abs(val) >= 1e3:
                    return f"{val/1e3:.1f}K"
                elif isinstance(val, float):
                    return f"{val:.2f}"
            return val
        
        # Apply formatting
        formatted_df = display_df.copy()
        for col in formatted_df.columns:
            if col not in ['Country', 'Continent']:
                try:
                    formatted_df[col] = formatted_df[col].apply(format_numbers)
                except:
                    pass
        
        # Display table
        st.dataframe(
            formatted_df,
            use_container_width=True,
            height=600
        )
        
        # Row count
        st.caption(f"Showing {len(formatted_df)} of {len(df)} total rows")
    
    with tab2:
        st.markdown("###  Summary Statistics")
        
        if selected_columns:
            numeric_cols = [col for col in selected_columns if col in df.select_dtypes(include=[np.number]).columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Calculate statistics
            stats_df = df[numeric_cols].describe().T
            
            # Add additional statistics
            stats_df['Missing'] = df[numeric_cols].isnull().sum()
            stats_df['Missing %'] = (stats_df['Missing'] / len(df)) * 100
            stats_df['Zero Count'] = (df[numeric_cols] == 0).sum()
            
            # Format for display
            display_stats = stats_df.round(2)
            
            st.dataframe(
                display_stats,
                use_container_width=True
            )
            
            # Distribution visualization
            st.markdown("####  Distribution Overview")
            
            selected_stat_col = st.selectbox(
                "Select variable for distribution:",
                numeric_cols,
                index=min(2, len(numeric_cols)-1)
            )
            
            if selected_stat_col in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df,
                        x=selected_stat_col,
                        nbins=30,
                        title=f'Distribution of {selected_stat_col}',
                        labels={selected_stat_col: selected_stat_col.replace('_', ' ').title()}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    if 'Continent' in df.columns:
                        fig = px.box(
                            df,
                            x='Continent',
                            y=selected_stat_col,
                            title=f'{selected_stat_col} by Continent',
                            labels={selected_stat_col: selected_stat_col.replace('_', ' ').title()}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns selected for statistical analysis.")
    
    with tab3:
        st.markdown("###  Data Quality Report")
        
        # Data completeness
        st.markdown("####  Data Completeness")
        
        completeness = pd.DataFrame({
            'Column': df.columns,
            'Total Values': len(df),
            'Non-Null Values': df.notnull().sum(),
            'Null Values': df.isnull().sum(),
            'Completeness %': (df.notnull().sum() / len(df) * 100).round(1)
        })
        
        st.dataframe(
            completeness.sort_values('Completeness %', ascending=False),
            use_container_width=True
        )
        
        # Data types
        st.markdown("####  Data Types")
        
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Unique Values': df.nunique(),
            'Sample Values': df.head(3).apply(lambda x: ', '.join(x.dropna().astype(str).tolist()) if x.notna().any() else '')
        })
        
        st.dataframe(dtype_info, use_container_width=True)
        
        # Outlier detection
        st.markdown("####  Outlier Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox(
                "Select variable for outlier detection:",
                numeric_cols,
                index=min(2, len(numeric_cols)-1)
            )
            
            if outlier_col in df.columns:
                # Calculate outlier bounds using IQR
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Outliers", len(outliers))
                
                with col2:
                    st.metric("Lower Bound", f"{lower_bound:.2f}")
                
                with col3:
                    st.metric("Upper Bound", f"{upper_bound:.2f}")
                
                if len(outliers) > 0:
                    st.write("**Outlier Countries:**")
                    for _, row in outliers.head(10).iterrows():
                        if 'Country' in row:
                            st.write(f"• {row['Country']}: {row[outlier_col]:.2f}")
    
    with tab4:
        st.markdown("###  Export Data")
        
        # Export options
        export_format = st.radio(
            "Select export format:",
            ["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        # Export scope
        export_scope = st.radio(
            "Export scope:",
            ["Current filtered view", "Complete dataset"],
            horizontal=True
        )
        
        # File name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"migration_data_{timestamp}"
        filename = st.text_input("File name:", value=default_filename)
        
        # Prepare data for export
        if export_scope == "Current filtered view":
            export_df = display_df if 'display_df' in locals() else df[selected_columns] if selected_columns else df
        else:
            export_df = df
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if export_format == "CSV":
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label=" Download CSV",
                    data=csv,
                    file_name=f"{filename}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if export_format == "Excel":
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, index=False, sheet_name='Migration Data')
                
                st.download_button(
                    label=" Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"{filename}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
        
        with col3:
            if export_format == "JSON":
                json_str = export_df.to_json(orient='records', indent=2)
                st.download_button(
                    label=" Download JSON",
                    data=json_str,
                    file_name=f"{filename}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Export statistics
        st.markdown("####  Export Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", len(export_df))
        
        with col2:
            st.metric("Columns", len(export_df.columns))
        
        with col3:
            file_size = len(str(export_df).encode('utf-8')) / 1024  # KB
            st.metric("Estimated Size", f"{file_size:.1f} KB")

else:
    st.error("Data not loaded. Please run the Jupyter notebook first.")
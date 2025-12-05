# app.py - Main entry point with multipage navigation

import streamlit as st
import pandas as pd
import plotly.express as px
import time

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Global Migration Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the entire app
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3B82F6;
    }
    
    /* Page navigation */
    .page-link {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .page-link:hover {
        background-color: #EFF6FF;
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Card content */
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        color: #64748B;
        font-size: 0.95rem;
        line-height: 1.4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'redirecting' not in st.session_state:
    st.session_state.redirecting = False

def show_home_page():
    """Main landing page for the app"""
    
    # Header section
    st.markdown('<h1 class="main-header"> Global Migration Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        <div style="
            color: #f8fafc;
            font-size: 1.2rem;
            line-height: 1.7;
            font-weight: 700;
            background: linear-gradient(135deg, #4b5563 0%, #374151 100%);
            padding: 1.5rem 1.8rem;
            border-radius: 0.75rem;
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
            border-left: 5px solid #3B82F6;
        ">
        A comprehensive platform for analyzing global migration patterns,
        population dynamics, and demographic indicators across 233 countries
        and territories. Explore interactive visualizations, insights,
        and predictive analytics.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Countries", "233", delta="+5 from 2024")
    
    with col3:
        st.metric("Data Points", "2,000+", delta="Updated Daily")
    
    st.markdown("---")
    
    # Page descriptions in a grid layout
    st.markdown("###  Explore Our Analysis Modules")
    
    # Create a grid of page cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Dashboard Overview</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Interactive dashboard with key metrics, filters, and global migration insights across all countries.</div>', unsafe_allow_html=True)
            
            if st.button("Explore Dashboard", key="btn1", use_container_width=True, type="primary"):
                st.session_state.current_page = "overview"
                st.session_state.redirecting = True
                st.rerun()
    
    with col2:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Migration Flows</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Analyze migration patterns, corridors, and country-level flows with interactive maps and charts.</div>', unsafe_allow_html=True)
            
            if st.button("View Flows", key="btn2", use_container_width=True, type="primary"):
                st.session_state.current_page = "migration"
                st.session_state.redirecting = True
                st.rerun()
    
    with col3:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Trends Analysis</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Explore demographic trends, correlations, and regional patterns over time with advanced analytics.</div>', unsafe_allow_html=True)
            
            if st.button("Analyze Trends", key="btn3", use_container_width=True, type="primary"):
                st.session_state.current_page = "trends"
                st.session_state.redirecting = True
                st.rerun()
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Forecasting</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Predictive models and migration scenario simulations using advanced machine learning algorithms.</div>', unsafe_allow_html=True)
            
            if st.button("Run Forecasts", key="btn4", use_container_width=True, type="primary"):
                st.session_state.current_page = "forecasting"
                st.session_state.redirecting = True
                st.rerun()
    
    with col5:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Data Explorer</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Interactive data table with filtering, sorting, and export capabilities for raw data analysis.</div>', unsafe_allow_html=True)
            
            if st.button("Explore Data", key="btn5", use_container_width=True, type="primary"):
                st.session_state.current_page = "explorer"
                st.session_state.redirecting = True
                st.rerun()
    
    with col6:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Methodology</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Learn about our data sources, calculation methods, and analytical approaches.</div>', unsafe_allow_html=True)
            
            if st.button("View Methodology", key="btn6", use_container_width=True, type="secondary"):
                st.session_state.current_page = "methodology"
                st.session_state.redirecting = True
                st.rerun()
    
    # Quick Stats Section
    st.markdown("---")
    st.markdown("###  Quick Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-value">1.2M</div>
            <div class="metric-label">Highest Immigration</div>
            <div style="font-size: 0.8rem; color: #3B82F6; margin-top: 0.5rem;">United States 🇺🇸</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">+15% from 2024</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col2:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-value">495K</div>
            <div class="metric-label">Highest Emigration</div>
            <div style="font-size: 0.8rem; color: #EF4444; margin-top: 0.5rem;">India 🇮🇳</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">+8% from 2024</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col3:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-value">67.5%</div>
            <div class="metric-label">Avg Urbanization Rate</div>
            <div style="font-size: 0.8rem; color: #10B981; margin-top: 0.5rem;">Top Destinations</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">Growing trend</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col4:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div class="metric-value">38.5</div>
            <div class="metric-label">Avg Median Age</div>
            <div style="font-size: 0.8rem; color: #8B5CF6; margin-top: 0.5rem;">Migration Destinations</div>
            <div style="font-size: 0.7rem; color: #94A3B8; margin-top: 0.25rem;">Aging population</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent Updates Section
    st.markdown("---")
    st.markdown("###  Recent Updates")
    
    update_col1, update_col2 = st.columns(2)
    
    with update_col1:
        with st.container(border=True):
            st.markdown("####  New Features")
            st.markdown("""
            - **Interactive Choropleth Maps**: Visualize migration flows geographically
            - **Predictive Analytics Module**: Forecast migration trends up to 2030
            - **Real-time Data Updates**: Latest population statistics integrated
            - **Export Capabilities**: Download data in CSV, Excel, and JSON formats
            """)
    
    with update_col2:
        with st.container(border=True):
            st.markdown("####  Latest Insights")
            st.markdown("""
            - **US immigration increased** by 15% in 2024
            - **India remains top source** of global migrants
            - **Urban migration accelerating** in developing nations
            - **Median age rising** in major destination countries
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center; 
        color: #64748B; 
        padding: 1.5rem;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 0.5rem;
        margin-top: 2rem;
    ">
    <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
    <strong>Data Sources:</strong> World Population Review 2025 • UN Migration Report • World Bank
    </div>
    <div style="font-size: 0.8rem;">
    Analysis Period: 2023-2025 • Last Updated: December 2024
    </div>
    </div>
    """, unsafe_allow_html=True)

def show_redirect_message(page_name):
    """Show redirecting message"""
    progress_container = st.empty()
    message_container = st.empty()
    
    with progress_container:
        st.progress(0)
    
    with message_container:
        st.info(f" Redirecting to {page_name}...")
    
    # Simulate loading progress
    for i in range(1, 101, 20):
        progress_container.progress(i / 100)
        time.sleep(0.1)
    
    progress_container.empty()
    message_container.empty()
    
    # Clear redirecting flag
    st.session_state.redirecting = False

def main():
    """Main application controller"""
    
    # Check if we're redirecting
    if st.session_state.get('redirecting', False):
        page_names = {
            "overview": "Dashboard Overview",
            "migration": "Migration Flows", 
            "trends": "Trends Analysis",
            "forecasting": "Forecasting",
            "explorer": "Data Explorer",
            "methodology": "Methodology"
        }
        
        current_page = st.session_state.current_page
        show_redirect_message(page_names.get(current_page, "the selected page"))
    
    # Handle page navigation
    if st.session_state.current_page != 'home':
        # In a real implementation, you would load the actual page content here
        # For now, we'll show a placeholder and option to return home
        
        container = st.container()
        with container:
            st.markdown(f"# {st.session_state.current_page.title()} Page")
            st.markdown("---")
            
            # Show page-specific content
            if st.session_state.current_page == "overview":
                st.markdown("""
                ###  Dashboard Overview
                This is where the interactive dashboard would be displayed.
                
                **Key Features:**
                - Real-time migration metrics
                - Interactive filters by region and country
                - Comparative analysis tools
                - Customizable visualizations
                """)
                
            elif st.session_state.current_page == "migration":
                st.markdown("""
                ###  Migration Flows
                This is where migration flow analysis would be displayed.
                
                **Key Features:**
                - Interactive world maps
                - Migration corridor visualization
                - Country-to-country flow analysis
                - Historical migration patterns
                """)
                
            elif st.session_state.current_page == "trends":
                st.markdown("""
                ###  Trends Analysis
                This is where trend analysis would be displayed.
                
                **Key Features:**
                - Time series analysis
                - Correlation matrices
                - Demographic trend visualization
                - Regional pattern identification
                """)
                
            elif st.session_state.current_page == "forecasting":
                st.markdown("""
                ### Forecasting
                This is where forecasting models would be displayed.
                
                **Key Features:**
                - Predictive analytics
                - Scenario simulations
                - Machine learning models
                - Future trend projections
                """)
                
            elif st.session_state.current_page == "explorer":
                st.markdown("""
                ###  Data Explorer
                This is where the data explorer would be displayed.
                
                **Key Features:**
                - Filterable data tables
                - Advanced search capabilities
                - Data export options
                - Raw data access
                """)
                
            elif st.session_state.current_page == "methodology":
                st.markdown("""
                ###  Methodology
                This is where methodology documentation would be displayed.
                
                **Key Features:**
                - Data collection methods
                - Calculation algorithms
                - Analytical approaches
                - Quality assurance processes
                """)
            
            st.markdown("---")
            
            # Return to home button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(" Return to Home", use_container_width=True, type="primary"):
                    st.session_state.current_page = 'home'
                    st.session_state.redirecting = True
                    st.rerun()
    else:
        # Show home page
        show_home_page()

if __name__ == "__main__":
    main()
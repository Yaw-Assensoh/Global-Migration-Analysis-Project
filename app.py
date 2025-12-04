# app.py - Main entry point with multipage navigation
import streamlit as st

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
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
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
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

def main():
    """Main landing page for the app"""
    
    # Header section
    st.markdown('<h1 class="main-header"> Global Migration Analytics Platform</h1>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        <div style="color: #475569; font-size: 1.1rem; line-height: 1.6;">
        A comprehensive platform for analyzing global migration patterns, 
        population dynamics, and demographic indicators across 233 countries 
        and territories. Explore interactive visualizations, insights, 
        and predictive analytics.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Countries", "233")
    
    with col3:
        st.metric("Data Points", "2,000+")
    
    st.markdown("---")
    
    # Page descriptions in a grid layout
    st.markdown("###  Explore Our Analysis Modules")
    
    # Create a grid of page cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size: 2rem;"></div>
        <h3>Dashboard Overview</h3>
        <p style="color: #64748B; font-size: 0.9rem;">
        Interactive dashboard with key metrics, filters, and global insights.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Overview", key="btn1", use_container_width=True):
            st.switch_page("pages/1_Overview.py")
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size: 2rem;"></div>
        <h3>Migration Flows</h3>
        <p style="color: #64748B; font-size: 0.9rem;">
        Analyze migration patterns, corridors, and country-level flows.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Migration Flows", key="btn2", use_container_width=True):
            st.switch_page("pages/2_Migration_Flows.py")
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size: 2rem;"></div>
        <h3>Trends Analysis</h3>
        <p style="color: #64748B; font-size: 0.9rem;">
        Explore demographic trends, correlations, and regional patterns.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Trends Analysis", key="btn3", use_container_width=True):
            st.switch_page("pages/3_Trends_Analysis.py")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size: 2rem;"></div>
        <h3>Forecasting</h3>
        <p style="color: #64748B; font-size: 0.9rem;">
        Predictive models and migration scenario simulations.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Forecasting", key="btn4", use_container_width=True):
            st.switch_page("pages/4_Forecasting.py")
    
    with col5:
        st.markdown("""
        <div class="metric-card">
        <div style="font-size: 2rem;"></div>
        <h3>Data Explorer</h3>
        <p style="color: #64748B; font-size: 0.9rem;">
        Interactive data table with filtering, sorting, and export.
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Data Explorer", key="btn5", use_container_width=True):
            st.switch_page("pages/5_Data_Explorer.py")
    
    # Quick Stats Section
    st.markdown("---")
    st.markdown("###  Quick Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">1.2M</div>
        <div class="metric-label">Highest Immigration</div>
        <div style="font-size: 0.8rem; color: #3B82F6;">United States</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">495K</div>
        <div class="metric-label">Highest Emigration</div>
        <div style="font-size: 0.8rem; color: #EF4444;">India</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">67.5%</div>
        <div class="metric-label">Avg Urbanization</div>
        <div style="font-size: 0.8rem; color: #10B981;">Top Migrant Destinations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value">38.5</div>
        <div class="metric-label">Avg Median Age</div>
        <div style="font-size: 0.8rem; color: #8B5CF6;">Migration Destinations</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748B; padding: 1rem;">
    <small>
    Data Source: World Population Review 2025 | Analysis Period: 2023-2025
    </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
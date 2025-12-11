# app.py - Main entry point with multipage navigation

import streamlit as st
import time

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Global Migration Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better graphics
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
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Enhanced page navigation */
    .page-link {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: white;
        border: 1px solid #E2E8F0;
    }
    
    .page-link:hover {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-color: #3B82F6;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Enhanced metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: #3B82F6;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
        text-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #64748B;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        transition: all 0.3s ease;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
    }
    
    /* Enhanced card content */
    .card-icon {
        font-size: 2.8rem;
        margin-bottom: 1rem;
        display: inline-block;
        padding: 0.8rem;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-radius: 12px;
        color: #3B82F6;
    }
    
    .card-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.75rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .card-description {
        color: #4B5563;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    
    /* Enhanced containers */
    .stContainer {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .stContainer:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-color: #3B82F6;
    }
    
    /* Enhanced HR */
    hr {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #3B82F6 50%, transparent 100%);
        border: none;
        margin: 2.5rem 0;
    }
    
    /* Enhanced badge styling */
    .country-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Glow effect for important elements */
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 5px rgba(59, 130, 246, 0.2);
        }
        to {
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
        }
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
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
    
    # Introduction - Enhanced with better graphics
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        <div style="
            color: #f8fafc;
            font-size: 1.2rem;
            line-height: 1.7;
            font-weight: 700;
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 8px 25px rgba(30, 58, 138, 0.3);
            border-left: 8px solid #8B5CF6;
            position: relative;
            overflow: hidden;
        ">
        <div style="position: absolute; top: -50px; right: -50px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
        <div style="position: absolute; bottom: -30px; left: -30px; width: 60px; height: 60px; background: rgba(255,255,255,0.1); border-radius: 50%;"></div>
        <div style="position: relative; z-index: 1;">
        A comprehensive platform for analyzing global migration patterns,
        population dynamics, and demographic indicators across 233 countries
        and territories. Explore interactive visualizations, insights,
        and predictive analytics.
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: bold; color: #1E3A8A;">233</div>
            <div style="font-size: 1rem; color: #64748B; margin-top: 0.5rem;">Countries</div>
            <div style="font-size: 0.85rem; color: #10B981; margin-top: 0.5rem;">+5 from 2024</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: bold; color: #1E3A8A;">2,000+</div>
            <div style="font-size: 1rem; color: #64748B; margin-top: 0.5rem;">Data Points</div>
            <div style="font-size: 0.85rem; color: #10B981; margin-top: 0.5rem;">Updated Daily</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Page descriptions in a grid layout
    st.markdown("### Explore Our Analysis Modules")
    
    # Create a grid of enhanced page cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon">📊</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Dashboard Overview</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Interactive dashboard with key metrics, filters, and global migration insights across all countries.</div>', unsafe_allow_html=True)
            
            if st.button("Explore Dashboard", key="btn1", use_container_width=True, type="primary"):
                st.session_state.current_page = "overview"
                st.session_state.redirecting = True
                st.rerun()
    
    with col2:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon">🗺️</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Migration Flows</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Analyze migration patterns, corridors, and country-level flows with interactive maps and charts.</div>', unsafe_allow_html=True)
            
            if st.button("View Flows", key="btn2", use_container_width=True, type="primary"):
                st.session_state.current_page = "migration"
                st.session_state.redirecting = True
                st.rerun()
    
    with col3:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon">📈</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="card-icon">🔮</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Forecasting</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Predictive models and migration scenario simulations using advanced machine learning algorithms.</div>', unsafe_allow_html=True)
            
            if st.button("Run Forecasts", key="btn4", use_container_width=True, type="primary"):
                st.session_state.current_page = "forecasting"
                st.session_state.redirecting = True
                st.rerun()
    
    with col5:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon">🔍</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Data Explorer</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Interactive data table with filtering, sorting, and export capabilities for raw data analysis.</div>', unsafe_allow_html=True)
            
            if st.button("Explore Data", key="btn5", use_container_width=True, type="primary"):
                st.session_state.current_page = "explorer"
                st.session_state.redirecting = True
                st.rerun()
    
    with col6:
        container = st.container(border=True)
        with container:
            st.markdown('<div class="card-icon">📚</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Methodology</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-description">Learn about our data sources, calculation methods, and analytical approaches.</div>', unsafe_allow_html=True)
            
            if st.button("View Methodology", key="btn6", use_container_width=True, type="secondary"):
                st.session_state.current_page = "methodology"
                st.session_state.redirecting = True
                st.rerun()
    
    # Quick Stats Section - Enhanced
    st.markdown("---")
    st.markdown("###  Quick Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div style="font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem;">1.2M</div>
            <div style="font-size: 0.95rem; color: #64748B; margin-bottom: 1rem; font-weight: 500;">Highest Immigration</div>
            <div style="background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; display: inline-block; margin: 0.5rem 0;">United States 🇺🇸</div>
            <div style="font-size: 0.85rem; color: #10B981; margin-top: 0.5rem; font-weight: 600;">+15% from 2024</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col2:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div style="font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem;">495K</div>
            <div style="font-size: 0.95rem; color: #64748B; margin-bottom: 1rem; font-weight: 500;">Highest Emigration</div>
            <div style="background: linear-gradient(135deg, #EF4444 0%, #F97316 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; display: inline-block; margin: 0.5rem 0;">India 🇮🇳</div>
            <div style="font-size: 0.85rem; color: #10B981; margin-top: 0.5rem; font-weight: 600;">+8% from 2024</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col3:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div style="font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem;">67.5%</div>
            <div style="font-size: 0.95rem; color: #64748B; margin-bottom: 1rem; font-weight: 500;">Avg Urbanization Rate</div>
            <div style="background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; display: inline-block; margin: 0.5rem 0;">Top Destinations</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem; font-weight: 500;">Growing trend</div>
            </div>
            """, unsafe_allow_html=True)
    
    with stats_col4:
        container = st.container(border=True)
        with container:
            st.markdown("""
            <div class="metric-card">
            <div style="font-size: 2.2rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem;">38.5</div>
            <div style="font-size: 0.95rem; color: #64748B; margin-bottom: 1rem; font-weight: 500;">Avg Median Age</div>
            <div style="background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%); color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; font-weight: 600; display: inline-block; margin: 0.5rem 0;">Migration Destinations</div>
            <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem; font-weight: 500;">Aging population</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent Updates Section - Enhanced
    st.markdown("---")
    st.markdown("###  Recent Updates")
    
    update_col1, update_col2 = st.columns(2)
    
    with update_col1:
        with st.container(border=True):
            st.markdown("""
            <div style="padding: 0.5rem 0;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="width: 8px; height: 30px; background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); border-radius: 4px; margin-right: 1rem;"></div>
            <h3 style="margin: 0; color: #1E3A8A;">New Features</h3>
            </div>
            <ul style="padding-left: 1.5rem; color: #4B5563;">
            <li style="margin-bottom: 0.75rem;"><strong>Interactive Choropleth Maps</strong>: Visualize migration flows geographically</li>
            <li style="margin-bottom: 0.75rem;"><strong>Predictive Analytics Module</strong>: Forecast migration trends up to 2030</li>
            <li style="margin-bottom: 0.75rem;"><strong>Real-time Data Updates</strong>: Latest population statistics integrated</li>
            <li><strong>Export Capabilities</strong>: Download data in CSV, Excel, and JSON formats</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with update_col2:
        with st.container(border=True):
            st.markdown("""
            <div style="padding: 0.5rem 0;">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="width: 8px; height: 30px; background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%); border-radius: 4px; margin-right: 1rem;"></div>
            <h3 style="margin: 0; color: #1E3A8A;">Latest Insights</h3>
            </div>
            <ul style="padding-left: 1.5rem; color: #4B5563;">
            <li style="margin-bottom: 0.75rem;"><strong>US immigration increased</strong> by 15% in 2024</li>
            <li style="margin-bottom: 0.75rem;"><strong>India remains top source</strong> of global migrants</li>
            <li style="margin-bottom: 0.75rem;"><strong>Urban migration accelerating</strong> in developing nations</li>
            <li><strong>Median age rising</strong> in major destination countries</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer - Enhanced
    st.markdown("---")
    st.markdown("""
    <div style="
        text-align: center; 
        color: #64748B; 
        padding: 2rem;
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border-radius: 16px;
        margin-top: 2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">
    <div style="font-size: 1rem; margin-bottom: 1rem; color: #1E3A8A; font-weight: 600;">
    <span style="display: inline-block; padding: 0.25rem 0.75rem; background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%); color: white; border-radius: 20px; margin: 0 0.5rem;">Data Sources</span>
    </div>
    <div style="font-size: 0.95rem; margin-bottom: 0.5rem;">
    <strong>World Population Review 2025</strong> • <strong>UN Migration Report</strong> • <strong>World Bank</strong>
    </div>
    <div style="font-size: 0.85rem; color: #94A3B8;">
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
                ###  Forecasting
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
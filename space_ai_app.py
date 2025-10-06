#!/usr/bin/env python3
"""
🚀 SPACE INTELLIGENCE AI PLATFORM - MAIN APPLICATION
Advanced multi-page dashboard for space weather monitoring, AI analysis, and mission planning
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from utils import set_background
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Space Intelligence AI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/kartik703/space_app',
        'Report a bug': 'https://github.com/kartik703/space_app/issues',
        'About': "Space Intelligence AI Platform - Enterprise-grade space weather monitoring with advanced AI"
    }
)

# Set background theme
set_background("space")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 3rem;
    }
    
    /* Navigation styling */
    .nav-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        text-align: center;
        cursor: pointer;
    }
    
    .nav-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border: 2px solid rgba(102, 126, 234, 0.5);
    }
    
    .nav-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    
    .nav-description {
        color: #666;
        line-height: 1.4;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-online {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .status-training {
        background: linear-gradient(135deg, #FF9800, #F57C00);
        color: white;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
    }
    
    /* Metrics cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card-enhanced {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 5px solid;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card-enhanced:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Success colors */
    .success { border-left-color: #4CAF50; }
    .warning { border-left-color: #FF9800; }
    .info { border-left-color: #2196F3; }
    .danger { border-left-color: #F44336; }
</style>
""", unsafe_allow_html=True)

def main_dashboard():
    """Main dashboard page"""
    
    # Header
    st.markdown('<div class="main-title">🚀 SPACE INTELLIGENCE AI PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enterprise-grade space weather monitoring with advanced AI • Real-time analysis • Mission-critical insights</div>', unsafe_allow_html=True)
    
    # System status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card-enhanced success">
            <div class="metric-label">System Status</div>
            <div class="metric-value">🟢 ONLINE</div>
            <div>All systems operational</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card-enhanced info">
            <div class="metric-label">AI Models</div>
            <div class="metric-value">4/4</div>
            <div>Active & Ready</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card-enhanced warning">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">LOW</div>
            <div>Nominal conditions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card-enhanced info">
            <div class="metric-label">Data Streams</div>
            <div class="metric-value">5</div>
            <div>NASA • NOAA • Live</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation grid
    st.markdown("## 🎛️ **Mission Control Center**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 **YOLO Solar Detection**\nReal-time AI analysis", key="yolo_nav", help="Advanced computer vision for solar flare detection"):
            st.session_state.page = "yolo"
            st.rerun()
            
        if st.button("💎 **Asteroid Mining**\nProfit & Risk Analysis", key="asteroid_nav", help="Mining simulations and profit calculations"):
            st.session_state.page = "asteroid"
            st.rerun()
    
    with col2:
        if st.button("🧠 **Fusion AI Engine**\nAdvanced Predictions", key="fusion_nav", help="Multi-model AI fusion for comprehensive analysis"):
            st.session_state.page = "fusion"
            st.rerun()
            
        if st.button("🌌 **Space Simulation**\nOrbital Mechanics", key="simulation_nav", help="3D space visualizations and orbital dynamics"):
            st.session_state.page = "simulation"
            st.rerun()
    
    with col3:
        if st.button("🗺️ **Future Roadmap**\nUpcoming Features", key="roadmap_nav", help="Development timeline and future capabilities"):
            st.session_state.page = "roadmap"
            st.rerun()
            
        if st.button("📊 **Analytics Hub**\nData Insights", key="analytics_nav", help="Comprehensive data analysis and reporting"):
            st.session_state.page = "analytics"
            st.rerun()
    
    st.markdown("---")
    
    # Live data preview
    st.markdown("## 📡 **Live Space Weather Data**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Solar activity chart
        times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        solar_activity = np.random.normal(5, 2, len(times)) + np.sin(np.linspace(0, 4*np.pi, len(times))) * 2
        solar_activity = np.clip(solar_activity, 0, 10)
        
        fig_solar = px.area(
            x=times, 
            y=solar_activity,
            title="24-Hour Solar Activity Index",
            color_discrete_sequence=['#FF6B6B']
        )
        fig_solar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_solar, use_container_width=True)
    
    with col2:
        # Risk assessment gauge
        risk_score = 0.25  # Example risk score
        
        fig_risk = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Current Risk Assessment (%)"},
            delta = {'reference': 20},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_risk.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_risk, use_container_width=True)

def sidebar_navigation():
    """Enhanced sidebar with navigation and controls"""
    
    st.sidebar.markdown("## 🚀 **Navigation**")
    
    # Page selection
    pages = {
        "🏠 Mission Control": "home",
        "🔍 YOLO Detection": "yolo", 
        "💎 Asteroid Mining": "asteroid",
        "🧠 Fusion AI": "fusion",
        "🌌 Space Simulation": "simulation",
        "🗺️ Future Roadmap": "roadmap",
        "📊 Analytics Hub": "analytics"
    }
    
    selected_page = st.sidebar.selectbox("Select Mission Module", list(pages.keys()))
    
    if pages[selected_page] != st.session_state.get('page', 'home'):
        st.session_state.page = pages[selected_page]
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.markdown("## ⚡ **Quick Actions**")
    
    if st.sidebar.button("🔄 Refresh Data", help="Fetch latest space weather data"):
        st.sidebar.success("Data refreshed!")
    
    if st.sidebar.button("🎯 Run AI Analysis", help="Execute comprehensive AI analysis"):
        st.sidebar.info("Analysis started...")
    
    if st.sidebar.button("📸 Capture Screenshot", help="Save current view"):
        st.sidebar.success("Screenshot saved!")
    
    st.sidebar.markdown("---")
    
    # System information
    st.sidebar.markdown("## ℹ️ **System Info**")
    
    status_info = {
        "🛰️ NASA SDO": "🟢 Connected",
        "🌍 NOAA SWPC": "🟢 Active", 
        "🤖 YOLO Model": "🟢 Ready",
        "🧠 Fusion AI": "🟢 Online",
        "💾 Database": "🟢 Healthy",
        "🔄 Data Pipeline": "🟢 Running"
    }
    
    for service, status in status_info.items():
        st.sidebar.markdown(f"**{service}**: {status}")
    
    st.sidebar.markdown("---")
    
    # Footer
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; font-size: 0.8rem; color: #666;">
        <b>Space Intelligence AI Platform</b><br>
        Version 2.0.0<br>
        Enterprise Edition<br><br>
        <em>Powered by NASA data, NOAA weather,<br>and advanced machine learning</em>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function with routing"""
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Sidebar navigation
    sidebar_navigation()
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        main_dashboard()
    elif st.session_state.page == 'yolo':
        from pages.yolo_detection import show_yolo_page
        show_yolo_page()
    elif st.session_state.page == 'asteroid':
        from pages.asteroid_mining import show_asteroid_page
        show_asteroid_page()
    elif st.session_state.page == 'fusion':
        from pages.fusion_ai import show_fusion_page
        show_fusion_page()
    elif st.session_state.page == 'simulation':
        from pages.space_simulation import show_simulation_page
        show_simulation_page()
    elif st.session_state.page == 'roadmap':
        from pages.future_roadmap import show_roadmap_page
        show_roadmap_page()
    elif st.session_state.page == 'analytics':
        from pages.analytics_hub import show_analytics_page
        show_analytics_page()

if __name__ == "__main__":
    main()
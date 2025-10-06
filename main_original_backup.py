"""
🚀 SPACE INTELLIGENCE PLATFORM - ALL-IN-ONE DASHBOARD
Advanced space mission control center with integrated AI, mining, simulation, and analytics
"""

import streamlit as st
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import networkx as nx
import random
import importlib
import traceback

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import core utilities
try:
    from core_utils import (
        apply_custom_theme, create_metric_card, create_status_indicator, 
        create_holographic_button, create_3d_earth, apply_space_theme_to_fig,
        generate_space_weather_data, generate_satellite_data
    )
except ImportError as e:
    st.error(f"Failed to import core utilities: {e}")
    st.stop()

# ======================== CONFIGURATION ========================

st.set_page_config(
    page_title="🚀 Space Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/space-intelligence/platform',
        'Report a bug': 'https://github.com/space-intelligence/platform/issues',
        'About': """
        # Space Intelligence Platform
        Advanced AI-powered space mission control and analytics system.
        
        **Features:**
        - Real-time space weather monitoring
        - 3D Earth and satellite visualization
        - YOLO-based solar flare detection
        - Asteroid mining analysis
        - Space mission simulation
        - AI fusion analytics
        
        **Version:** 2.0.0
        **Built with:** Streamlit, Plotly, PyTorch, OpenCV
        """
    }
)

# ======================== PAGE MODULES ========================

PAGES = {
    "🌍 Space Command Center": {
        "module": "pages.command_center",
        "function": "show_command_center",
        "description": "Mission control dashboard with real-time space data",
        "icon": "🌍",
        "category": "Core"
    },
    "🔥 YOLO Detection System": {
        "module": "pages.yolo_system", 
        "function": "show_yolo_system",
        "description": "AI-powered solar flare and space anomaly detection",
        "icon": "🔥",
        "category": "AI Systems"
    },
    "⛏️ Asteroid Mining Platform": {
        "module": "pages.asteroid_mining",
        "function": "show_asteroid_mining", 
        "description": "Asteroid analysis and mining opportunity assessment",
        "icon": "⛏️",
        "category": "Mining"
    },
    "🌌 Space Simulation Center": {
        "module": "pages.space_simulation",
        "function": "show_space_simulation",
        "description": "3D orbital mechanics and mission trajectory planning",
        "icon": "🌌", 
        "category": "Simulation"
    },
    "🤖 Fusion AI Engine": {
        "module": "pages.fusion_ai",
        "function": "show_fusion_ai",
        "description": "Multi-model AI fusion for advanced space analytics", 
        "icon": "🤖",
        "category": "AI Systems"
    },
    "📊 Analytics Hub": {
        "module": "pages.analytics_hub",
        "function": "show_analytics_hub",
        "description": "Comprehensive data analysis and reporting center",
        "icon": "📊",
        "category": "Analytics"
    }
}

# ======================== APPLICATION STATE ========================

def initialize_session_state():
    """Initialize session state variables"""
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "🌍 Space Command Center"
    
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'theme': 'dark',
            'auto_refresh': True,
            'refresh_interval': 30,
            'notifications': True,
            'sound_alerts': False
        }
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'ai_systems': 'online',
            'data_pipeline': 'online', 
            'space_weather': 'online',
            'satellite_tracking': 'online',
            'mining_analysis': 'online'
        }

# ======================== SIDEBAR NAVIGATION ========================

def render_sidebar():
    """Render the sidebar navigation with system status"""
    
    with st.sidebar:
        # Platform Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
            <h1 style="font-family: 'Orbitron', sans-serif; font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, #00D4FF, #9D4EDD); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🚀 SPACE<br>INTELLIGENCE
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #B0BEC5; font-size: 0.9rem;">Mission Control Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status Overview
        st.markdown("### 🔋 System Status")
        
        status_items = [
            ("AI Systems", st.session_state.system_status['ai_systems']),
            ("Data Pipeline", st.session_state.system_status['data_pipeline']),
            ("Space Weather", st.session_state.system_status['space_weather']),
            ("Satellites", st.session_state.system_status['satellite_tracking'])
        ]
        
        for label, status in status_items:
            create_status_indicator(status, label)
        
        st.markdown("---")
        
        # Navigation Menu
        st.markdown("### 🧭 Navigation")
        
        # Group pages by category
        categories = {}
        for page_name, page_info in PAGES.items():
            category = page_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((page_name, page_info))
        
        # Render navigation by category
        for category, pages in categories.items():
            st.markdown(f"**{category}**")
            
            for page_name, page_info in pages:
                # Create navigation button
                button_style = "nav-button"
                if st.session_state.current_page == page_name:
                    button_style += " active"
                
                if st.button(
                    f"{page_info['icon']} {page_name.split(' ', 1)[1]}",
                    key=f"nav_{page_name}",
                    help=page_info['description']
                ):
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if create_holographic_button("🔄 Refresh", key="refresh_all"):
                st.success("Systems refreshed!")
                st.rerun()
        
        with col2:
            if create_holographic_button("⚙️ Settings", key="settings"):
                show_settings_modal()
        
        # System Information
        st.markdown("---")
        st.markdown("### 📡 Live Data")
        
        current_time = datetime.now()
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; font-family: 'Space Mono', monospace;">
            <strong>Mission Time:</strong><br>
            {current_time.strftime('%Y-%m-%d')}<br>
            {current_time.strftime('%H:%M:%S UTC')}
        </div>
        """, unsafe_allow_html=True)
        
        # Performance Metrics
        st.markdown("### 📊 Performance")
        
        metrics = [
            ("CPU", "67%", "#4CAF50"),
            ("Memory", "2.1GB", "#FF9800"), 
            ("Network", "98%", "#4CAF50"),
            ("Storage", "45%", "#4CAF50")
        ]
        
        for metric, value, color in metrics:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; margin: 0.3rem 0; color: white;">
                <span>{metric}:</span>
                <span style="color: {color}; font-weight: bold;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

def show_settings_modal():
    """Show settings configuration modal"""
    
    with st.expander("⚙️ Platform Settings", expanded=True):
        st.markdown("### User Preferences")
        
        # Theme selection
        theme = st.selectbox(
            "Interface Theme",
            ["Dark Space", "Light Mode", "High Contrast"],
            index=0
        )
        
        # Auto-refresh settings
        auto_refresh = st.checkbox("Auto-refresh data", value=st.session_state.user_settings['auto_refresh'])
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)",
                min_value=5,
                max_value=300,
                value=st.session_state.user_settings['refresh_interval']
            )
        
        # Notification settings
        st.markdown("### Notifications")
        notifications = st.checkbox("Enable notifications", value=st.session_state.user_settings['notifications'])
        sound_alerts = st.checkbox("Sound alerts", value=st.session_state.user_settings['sound_alerts'])
        
        # Data settings
        st.markdown("### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Data"):
                st.success("Data export initiated!")
        
        with col2:
            if st.button("🔄 Reset Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.session_state.user_settings.update({
                'theme': theme,
                'auto_refresh': auto_refresh,
                'refresh_interval': refresh_interval if auto_refresh else 30,
                'notifications': notifications,
                'sound_alerts': sound_alerts
            })
            st.success("Settings saved successfully!")

# ======================== PAGE LOADING ========================

def load_page_module(page_info):
    """Dynamically load and execute page module"""
    
    try:
        # Import the module
        module = importlib.import_module(page_info['module'])
        
        # Get the function
        page_function = getattr(module, page_info['function'])
        
        # Execute the function
        page_function()
        
    except ImportError as e:
        st.error(f"""
        **Module Import Error**
        
        Could not import module: `{page_info['module']}`
        
        **Error:** {str(e)}
        
        **Solution:** Ensure the page module exists and is properly configured.
        """)
        
        # Show fallback content
        show_fallback_page(page_info)
        
    except AttributeError as e:
        st.error(f"""
        **Function Not Found**
        
        Function `{page_info['function']}` not found in module `{page_info['module']}`
        
        **Error:** {str(e)}
        """)
        
        show_fallback_page(page_info)
        
    except Exception as e:
        st.error(f"""
        **Runtime Error**
        
        An error occurred while loading the page.
        
        **Error:** {str(e)}
        
        **Traceback:**
        ```
        {traceback.format_exc()}
        ```
        """)
        
        show_fallback_page(page_info)

def show_fallback_page(page_info):
    """Show fallback content when page loading fails"""
    
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 2rem; background: rgba(255,255,255,0.05); border-radius: 20px; margin: 2rem 0;">
        <h1 style="font-size: 4rem; margin: 0;">{page_info['icon']}</h1>
        <h2 style="color: #00D4FF; font-family: 'Orbitron', sans-serif;">Under Construction</h2>
        <p style="font-size: 1.2rem; color: #B0BEC5; margin: 1rem 0;">{page_info['description']}</p>
        <p style="color: #FF9800;">This module is currently being developed and will be available soon.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Development status
    st.markdown("### 🚧 Development Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card("Design", "90%", color="green")
    
    with col2:
        create_metric_card("Implementation", "60%", color="gold")
    
    with col3:
        create_metric_card("Testing", "30%", color="red")
    
    # Feature preview
    st.markdown("### 🎯 Planned Features")
    
    features = [
        "Real-time data visualization",
        "Interactive 3D interfaces", 
        "AI-powered analytics",
        "Advanced filtering and search",
        "Export and reporting capabilities",
        "Mobile-responsive design"
    ]
    
    for i, feature in enumerate(features, 1):
        st.markdown(f"**{i}.** {feature}")

# ======================== MAIN APPLICATION ========================

def main():
    """Main application entry point"""
    
    # Apply custom theme
    apply_custom_theme()
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar navigation
    render_sidebar()
    
    # Main content area
    current_page = st.session_state.current_page
    
    if current_page in PAGES:
        # Load the selected page
        page_info = PAGES[current_page]
        
        # Page header
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; font-family: 'Orbitron', sans-serif; background: linear-gradient(135deg, #00D4FF, #9D4EDD); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                {page_info['icon']} {current_page}
            </h1>
            <p style="font-size: 1.2rem; color: #B0BEC5; margin: 0;">{page_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load page content
        load_page_module(page_info)
    
    else:
        st.error(f"Page '{current_page}' not found!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: #666;">
        <p style="margin: 0; font-family: 'Space Mono', monospace;">
            🚀 Space Intelligence Platform v2.0.0 | 
            Built with ❤️ for space exploration | 
            © 2024 Space Intelligence Labs
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================== RUN APPLICATION ========================

if __name__ == "__main__":
    main()
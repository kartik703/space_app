#!/usr/bin/env python3
"""
🚀 STREAMLIT CLOUD DEPLOYMENT ENTRY POINT
Space Intelligence Platform - Comprehensive Dashboard
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main entry point for Streamlit Cloud"""
    
    # Configure page
    st.set_page_config(
        page_title="🚀 Space Intelligence Platform",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import and run your comprehensive dashboard
    try:
        # Try to import your comprehensive main dashboard
        import main as comprehensive_main
        comprehensive_main.main()
        
    except ImportError as e:
        # Fallback error handling
        st.error(f"❌ Failed to load comprehensive dashboard: {e}")
        
        # Show basic info
        st.title("🚀 Space Intelligence Platform")
        st.markdown("""
        ## 🌟 Features:
        - 🤖 **YOLO Detection** - Real-time solar activity analysis
        - 🌌 **Simulations** - Space mission control & orbital mechanics  
        - 💎 **Asteroid Data** - Mining calculations & profit analysis
        - 🛰️ **Live Data Feeds** - NASA, NOAA, ISS integration
        - ⚡ **Space Weather** - Live monitoring & forecasting
        - 🔍 **AI Analytics** - Computer vision & ML models
        
        ## 🚀 Repository:
        [GitHub - Space Intelligence Platform](https://github.com/kartik703/space_app)
        """)
        
        st.error("Please check deployment logs and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()
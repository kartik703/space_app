#!/usr/bin/env python3
"""
🚀 SPACE AI SYSTEM - Main Application Entry Point
Real-time space weather monitoring with AI-powered analysis
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import the ALL-IN-ONE comprehensive dashboard with simulations, YOLO, asteroids, and all data
try:
    import main as comprehensive_main
    COMPREHENSIVE_AVAILABLE = True
except ImportError:
    try:
        from space_ai_app import main as space_ai_main
        SPACE_AI_AVAILABLE = True
        COMPREHENSIVE_AVAILABLE = False
    except ImportError:
        SPACE_AI_AVAILABLE = False
        COMPREHENSIVE_AVAILABLE = False

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="🚀 Space AI System",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check for the ALL-IN-ONE comprehensive dashboard
    if COMPREHENSIVE_AVAILABLE:
        # Run the ALL-IN-ONE dashboard with simulations, YOLO data, asteroid data, and everything
        comprehensive_main.main()
    elif SPACE_AI_AVAILABLE:
        # Fallback to space AI dashboard
        space_ai_main()
    else:
        # Fallback interface
        st.title("🚀 Space AI System")
        st.error("❌ ALL-IN-ONE comprehensive dashboard not available. Please check installation.")
        
        st.markdown("""
        ## Quick Setup:
        
        1. **Install dependencies:**
        ```bash
        pip install -r requirements.txt
        ```
        
        2. **Launch system:**
        ```bash
        python launch.py
        ```
        
        3. **Or run dashboard directly:**
        ```bash
        streamlit run fusion_ai_live.py
        ```
        """)

if __name__ == "__main__":
    main()
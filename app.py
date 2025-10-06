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

# Import the main dashboard
try:
    from fusion_ai_live import main as fusion_main
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="🚀 Space AI System",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check if this is the main dashboard or redirect
    if FUSION_AVAILABLE:
        # Run the fusion AI dashboard
        fusion_main()
    else:
        # Fallback interface
        st.title("🚀 Space AI System")
        st.error("❌ Main dashboard not available. Please check installation.")
        
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
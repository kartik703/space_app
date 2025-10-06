"""
🚀 SPACE INTELLIGENCE PLATFORM - CORE UTILITIES
Advanced utilities for 3D visualizations, space calculations, and UI components
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import json

# ======================== STYLING & THEMES ========================

def apply_custom_theme():
    """Apply comprehensive custom theme with 3D space aesthetics"""
    
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Mono:wght@400;700&display=swap');
    
    /* Global Variables */
    :root {
        --primary-bg: #0B1426;
        --secondary-bg: #1A2332;
        --accent-blue: #00D4FF;
        --accent-purple: #9D4EDD;
        --accent-gold: #FFD700;
        --text-primary: #FFFFFF;
        --text-secondary: #B0BEC5;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --danger-color: #F44336;
    }
    
    /* Main Background with Animation */
    .stApp {
        background: linear-gradient(135deg, #0B1426 0%, #1A2332 50%, #2D1B69 100%);
        background-attachment: fixed;
        font-family: 'Space Mono', monospace;
    }
    
    /* Animated Stars Background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, #fff, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #fff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #fff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: sparkle 20s linear infinite;
        z-index: -1;
        opacity: 0.3;
    }
    
    @keyframes sparkle {
        from {transform: translateY(0px);}
        to {transform: translateY(-100px);}
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(13, 20, 38, 0.95) 0%, rgba(26, 35, 50, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-right: 2px solid var(--accent-blue);
    }
    
    /* Navigation Buttons */
    .nav-button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(157, 78, 221, 0.2));
        border: 1px solid var(--accent-blue);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.4), rgba(157, 78, 221, 0.4));
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Headers */
    h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
    }
    
    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: var(--text-primary);
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 212, 255, 0.2);
    }
    
    /* Glowing Elements */
    .glow-blue {
        box-shadow: 0 0 20px var(--accent-blue);
        border: 1px solid var(--accent-blue);
    }
    
    .glow-purple {
        box-shadow: 0 0 20px var(--accent-purple);
        border: 1px solid var(--accent-purple);
    }
    
    .glow-gold {
        box-shadow: 0 0 20px var(--accent-gold);
        border: 1px solid var(--accent-gold);
    }
    
    /* Status Indicators */
    .status-online {
        color: var(--success-color);
        text-shadow: 0 0 10px var(--success-color);
    }
    
    .status-warning {
        color: var(--warning-color);
        text-shadow: 0 0 10px var(--warning-color);
    }
    
    .status-critical {
        color: var(--danger-color);
        text-shadow: 0 0 10px var(--danger-color);
    }
    
    /* Holographic Effect */
    .holographic {
        background: linear-gradient(45deg, 
            rgba(0, 212, 255, 0.1) 0%,
            rgba(157, 78, 221, 0.1) 25%,
            rgba(255, 215, 0, 0.1) 50%,
            rgba(157, 78, 221, 0.1) 75%,
            rgba(0, 212, 255, 0.1) 100%);
        background-size: 400% 400%;
        animation: hologram 4s ease infinite;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    @keyframes hologram {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* 3D Button Effect */
    .button-3d {
        background: linear-gradient(145deg, #2D1B69, #1A2332);
        box-shadow: 
            inset 5px 5px 10px rgba(0, 0, 0, 0.3),
            inset -5px -5px 10px rgba(255, 255, 255, 0.1),
            0 10px 20px rgba(0, 212, 255, 0.2);
        border-radius: 15px;
        padding: 1rem 2rem;
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .button-3d:hover {
        transform: translateY(-3px);
        box-shadow: 
            inset 5px 5px 10px rgba(0, 0, 0, 0.3),
            inset -5px -5px 10px rgba(255, 255, 255, 0.1),
            0 15px 30px rgba(0, 212, 255, 0.4);
    }
    
    .button-3d:active {
        transform: translateY(0px);
        box-shadow: 
            inset 5px 5px 10px rgba(0, 0, 0, 0.5),
            inset -5px -5px 10px rgba(255, 255, 255, 0.05),
            0 5px 10px rgba(0, 212, 255, 0.2);
    }
    
    /* Data Tables */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--accent-purple), var(--accent-gold));
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid rgba(0, 212, 255, 0.3);
        border-top: 4px solid var(--accent-blue);
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Particle System */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        width: 2px;
        height: 2px;
        background: var(--accent-blue);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 1; }
        50% { transform: translateY(-20px) rotate(180deg); opacity: 0.5; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

# ======================== 3D VISUALIZATION UTILITIES ========================

def create_3d_earth(show_satellites=True, show_atmosphere=True):
    """Create realistic 3D Earth visualization with satellites"""
    
    # Earth sphere coordinates
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    earth_radius = 6371  # km
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    # Earth surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[
            [0, '#1e3c72'],     # Ocean deep
            [0.3, '#2a5298'],   # Ocean
            [0.5, '#228b22'],   # Land
            [0.7, '#32cd32'],   # Forest
            [0.8, '#daa520'],   # Desert
            [1, '#ffffff']      # Ice caps
        ],
        showscale=False,
        name='Earth',
        hovertemplate='Earth Surface<extra></extra>'
    ))
    
    # Atmosphere glow effect
    if show_atmosphere:
        atm_radius = earth_radius * 1.02
        x_atm = atm_radius * np.outer(np.cos(u), np.sin(v))
        y_atm = atm_radius * np.outer(np.sin(u), np.sin(v))
        z_atm = atm_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_atm, y=y_atm, z=z_atm,
            colorscale=[[0, 'rgba(135,206,250,0.3)'], [1, 'rgba(135,206,250,0.1)']],
            showscale=False,
            name='Atmosphere',
            opacity=0.3,
            hoverinfo='skip'
        ))
    
    # Add satellites if requested
    if show_satellites:
        # Generate satellite positions
        n_satellites = 20
        np.random.seed(42)
        
        sat_distances = np.random.uniform(earth_radius + 400, earth_radius + 35786, n_satellites)
        sat_theta = np.random.uniform(0, 2*np.pi, n_satellites)
        sat_phi = np.random.uniform(0, np.pi, n_satellites)
        
        sat_x = sat_distances * np.sin(sat_phi) * np.cos(sat_theta)
        sat_y = sat_distances * np.sin(sat_phi) * np.sin(sat_theta)
        sat_z = sat_distances * np.cos(sat_phi)
        
        # Satellite types
        sat_types = np.random.choice(['GPS', 'Communication', 'Weather', 'Research'], n_satellites)
        colors = {'GPS': '#FFD700', 'Communication': '#00D4FF', 'Weather': '#32CD32', 'Research': '#FF69B4'}
        
        for i, sat_type in enumerate(set(sat_types)):
            mask = sat_types == sat_type
            fig.add_trace(go.Scatter3d(
                x=sat_x[mask],
                y=sat_y[mask],
                z=sat_z[mask],
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors[sat_type],
                    symbol='diamond'
                ),
                name=f'{sat_type} Satellites',
                hovertemplate=f'{sat_type} Satellite<br>Altitude: %{{customdata}} km<extra></extra>',
                customdata=sat_distances[mask] - earth_radius
            ))
    
    # Layout for 3D scene
    fig.update_layout(
        title=dict(
            text="🌍 Real-time Earth Visualization",
            font=dict(size=24, family="Orbitron, sans-serif"),
            x=0.5
        ),
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)", 
            zaxis_title="Z (km)",
            bgcolor="black",
            xaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            yaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            zaxis=dict(
                backgroundcolor="black",
                gridcolor="rgba(255,255,255,0.1)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.2)"
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=600
    )
    
    return fig

def create_space_trajectory(departure_body, target_body, trajectory_type="Hohmann"):
    """Create 3D space trajectory visualization"""
    
    # Body positions (simplified circular orbits)
    bodies = {
        'Earth': {'distance': 1.0, 'color': '#6B93D6', 'size': 8},
        'Mars': {'distance': 1.52, 'color': '#CD5C5C', 'size': 6},
        'Jupiter': {'distance': 5.20, 'color': '#D8CA9D', 'size': 20},
        'Moon': {'distance': 0.00257, 'color': '#C0C0C0', 'size': 4},  # Moon orbit around Earth
        'Venus': {'distance': 0.72, 'color': '#FFC649', 'size': 7}
    }
    
    fig = go.Figure()
    
    # Add Sun
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=25, color='#FFD700'),
        name='Sun'
    ))
    
    # Add planetary orbits and positions
    for body, props in bodies.items():
        if body == 'Moon' and departure_body != 'Earth' and target_body != 'Moon':
            continue
            
        # Orbit path
        theta = np.linspace(0, 2*np.pi, 100)
        orbit_x = props['distance'] * np.cos(theta)
        orbit_y = props['distance'] * np.sin(theta)
        orbit_z = np.zeros_like(orbit_x)
        
        fig.add_trace(go.Scatter3d(
            x=orbit_x, y=orbit_y, z=orbit_z,
            mode='lines',
            line=dict(color=props['color'], width=2),
            name=f'{body} Orbit',
            showlegend=False
        ))
        
        # Current position (arbitrary)
        current_angle = np.pi/4 if body == departure_body else np.pi/2
        pos_x = props['distance'] * np.cos(current_angle)
        pos_y = props['distance'] * np.sin(current_angle)
        
        fig.add_trace(go.Scatter3d(
            x=[pos_x], y=[pos_y], z=[0],
            mode='markers',
            marker=dict(size=props['size'], color=props['color']),
            name=body
        ))
    
    # Add trajectory based on type
    if trajectory_type == "Hohmann":
        # Hohmann transfer orbit
        dep_dist = bodies[departure_body]['distance']
        tgt_dist = bodies[target_body]['distance']
        
        # Transfer orbit parameters
        a_transfer = (dep_dist + tgt_dist) / 2
        
        # Create transfer trajectory
        theta_transfer = np.linspace(0, np.pi, 50)
        r_transfer = a_transfer * (1 - 0.1**2) / (1 + 0.1 * np.cos(theta_transfer))  # Small eccentricity
        
        transfer_x = r_transfer * np.cos(theta_transfer)
        transfer_y = r_transfer * np.sin(theta_transfer)
        transfer_z = np.zeros_like(transfer_x)
        
        fig.add_trace(go.Scatter3d(
            x=transfer_x, y=transfer_y, z=transfer_z,
            mode='lines',
            line=dict(color='white', width=4, dash='dash'),
            name=f'{departure_body}-{target_body} Transfer'
        ))
    
    fig.update_layout(
        title=f"🚀 {departure_body} to {target_body} Mission Trajectory",
        scene=dict(
            xaxis_title="X (AU)",
            yaxis_title="Y (AU)",
            zaxis_title="Z (AU)",
            bgcolor="black",
            aspectmode='cube'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

# ======================== SPACE CALCULATIONS ========================

def calculate_orbital_parameters(altitude_km, eccentricity=0, inclination_deg=0):
    """Calculate orbital mechanics parameters"""
    
    earth_radius = 6371  # km
    earth_mu = 398600.4418  # km³/s²
    
    # Semi-major axis
    a = earth_radius + altitude_km
    
    # Orbital period (Kepler's 3rd law)
    T = 2 * np.pi * np.sqrt(a**3 / earth_mu)  # seconds
    
    # Orbital velocity
    v = np.sqrt(earth_mu / a)  # km/s
    
    # Escape velocity
    v_escape = np.sqrt(2 * earth_mu / a)  # km/s
    
    return {
        'semi_major_axis': a,
        'period_seconds': T,
        'period_minutes': T / 60,
        'period_hours': T / 3600,
        'orbital_velocity': v,
        'escape_velocity': v_escape,
        'eccentricity': eccentricity,
        'inclination': inclination_deg
    }

def calculate_hohmann_transfer(r1_au, r2_au):
    """Calculate Hohmann transfer parameters between two circular orbits"""
    
    # Transfer orbit semi-major axis
    a_transfer = (r1_au + r2_au) / 2
    
    # Transfer time (half the transfer orbit period)
    transfer_time_years = np.sqrt(a_transfer**3) / 2
    transfer_time_days = transfer_time_years * 365.25
    
    # Velocity changes needed
    # At departure (periapsis)
    v1_circular = np.sqrt(1 / r1_au)  # Normalized units
    v1_transfer = np.sqrt(2 * (1/r1_au - 1/(2*a_transfer)))
    delta_v1 = abs(v1_transfer - v1_circular)
    
    # At arrival (apoapsis)
    v2_circular = np.sqrt(1 / r2_au)
    v2_transfer = np.sqrt(2 * (1/r2_au - 1/(2*a_transfer)))
    delta_v2 = abs(v2_circular - v2_transfer)
    
    total_delta_v = delta_v1 + delta_v2
    
    return {
        'transfer_time_days': transfer_time_days,
        'delta_v_departure': delta_v1,
        'delta_v_arrival': delta_v2,
        'total_delta_v': total_delta_v,
        'transfer_semi_major_axis': a_transfer
    }

# ======================== DATA GENERATION ========================

def generate_space_weather_data():
    """Generate realistic space weather data"""
    
    current_time = datetime.now()
    times = [current_time - timedelta(hours=i) for i in range(24, 0, -1)]
    
    # Solar wind speed (km/s)
    base_speed = 400
    solar_wind = [base_speed + np.random.normal(0, 50) + 100*np.sin(i/4) for i in range(24)]
    
    # Kp index (geomagnetic activity)
    kp_values = [max(0, min(9, 3 + np.random.normal(0, 1) + 2*np.sin(i/6))) for i in range(24)]
    
    # Solar flux (SFU)
    solar_flux = [120 + np.random.normal(0, 20) + 30*np.cos(i/8) for i in range(24)]
    
    return pd.DataFrame({
        'timestamp': times,
        'solar_wind_speed': solar_wind,
        'kp_index': kp_values,
        'solar_flux': solar_flux
    })

def generate_satellite_data():
    """Generate realistic satellite tracking data"""
    
    satellites = []
    sat_types = ['GPS', 'Communication', 'Weather', 'Research', 'Military', 'Commercial']
    
    for i in range(50):
        # Generate 3D position coordinates
        altitude = np.random.uniform(200, 35786)  # km
        inclination = np.random.uniform(0, 180)   # degrees
        longitude = np.random.uniform(0, 360)     # degrees
        
        # Convert to 3D coordinates (simplified spherical to cartesian)
        earth_radius = 6371  # km
        total_radius = earth_radius + altitude
        
        # Convert spherical coordinates to cartesian
        inc_rad = np.radians(inclination)
        lon_rad = np.radians(longitude)
        
        x_pos = total_radius * np.sin(inc_rad) * np.cos(lon_rad) / 1000  # Scale down
        y_pos = total_radius * np.sin(inc_rad) * np.sin(lon_rad) / 1000
        z_pos = total_radius * np.cos(inc_rad) / 1000
        
        sat = {
            'id': f'SAT-{i+1:03d}',
            'name': f'{np.random.choice(sat_types)} Satellite {i+1}',
            'type': np.random.choice(sat_types),
            'altitude': altitude,
            'inclination': inclination,
            'longitude': longitude,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'z_pos': z_pos,
            'status': np.random.choice(['Active', 'Active', 'Active', 'Maintenance', 'Deorbited'], p=[0.7, 0.15, 0.1, 0.04, 0.01]),
            'last_contact': datetime.now() - timedelta(minutes=np.random.randint(0, 120)),
            'battery_level': np.random.uniform(60, 100),
            'signal_strength': np.random.uniform(-80, -40)  # dBm
        }
        satellites.append(sat)
    
    return pd.DataFrame(satellites)

def generate_asteroid_data():
    """Generate asteroid mining target data"""
    
    asteroids = []
    compositions = ['Metallic', 'Carbonaceous', 'Silicate', 'Mixed']
    
    for i in range(100):
        asteroid = {
            'designation': f'{2020 + i//10} {chr(65 + i%26)}{i%10}',
            'distance_au': np.random.uniform(1.2, 4.5),
            'diameter_km': np.random.lognormal(0, 1.5),
            'composition': np.random.choice(compositions),
            'estimated_value_billions': np.random.lognormal(2, 2),
            'accessibility_score': np.random.uniform(0.1, 1.0),
            'water_content_percent': np.random.uniform(0, 20),
            'metal_content_percent': np.random.uniform(5, 95),
            'mission_duration_years': np.random.uniform(2, 8),
            'delta_v_requirement': np.random.uniform(5, 15)  # km/s
        }
        asteroids.append(asteroid)
    
    return pd.DataFrame(asteroids)

# ======================== UI COMPONENTS ========================

def create_metric_card(title, value, unit="", delta=None, color="blue"):
    """Create beautiful metric card with 3D effects"""
    
    delta_html = ""
    if delta is not None:
        delta_color = "#4CAF50" if delta >= 0 else "#F44336"
        delta_symbol = "↗" if delta >= 0 else "↘"
        delta_html = f'<p style="color: {delta_color}; margin: 0; font-size: 0.9rem;">{delta_symbol} {abs(delta):.1f}%</p>'
    
    colors = {
        "blue": "linear-gradient(135deg, #00D4FF, #0099CC)",
        "purple": "linear-gradient(135deg, #9D4EDD, #7B2CBF)",
        "gold": "linear-gradient(135deg, #FFD700, #FFA500)",
        "green": "linear-gradient(135deg, #4CAF50, #45A049)",
        "red": "linear-gradient(135deg, #F44336, #D32F2F)"
    }
    
    st.markdown(f"""
    <div class="metric-card" style="background: {colors.get(color, colors['blue'])};">
        <h3 style="margin: 0; color: white; font-family: 'Orbitron', sans-serif;">{title}</h3>
        <h1 style="margin: 0.5rem 0; color: white; font-size: 2.5rem; font-family: 'Space Mono', monospace;">{value} <span style="font-size: 1rem;">{unit}</span></h1>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def create_status_indicator(status, label):
    """Create glowing status indicator"""
    
    status_configs = {
        "online": {"color": "#4CAF50", "icon": "🟢", "text": "ONLINE"},
        "warning": {"color": "#FF9800", "icon": "🟡", "text": "WARNING"},
        "critical": {"color": "#F44336", "icon": "🔴", "text": "CRITICAL"},
        "offline": {"color": "#757575", "icon": "⚫", "text": "OFFLINE"}
    }
    
    config = status_configs.get(status.lower(), status_configs["offline"])
    
    st.markdown(f"""
    <div style="
        display: flex; 
        align-items: center; 
        background: rgba(255,255,255,0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid {config['color']};
        margin: 0.5rem 0;
    ">
        <span style="font-size: 1.5rem; margin-right: 1rem;">{config['icon']}</span>
        <div>
            <h4 style="margin: 0; color: white;">{label}</h4>
            <p style="margin: 0; color: {config['color']}; font-weight: bold;">{config['text']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_loading_animation(text="Loading..."):
    """Create space-themed loading animation"""
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <p style="color: #00D4FF; font-family: 'Orbitron', sans-serif; margin-top: 1rem;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

def create_holographic_button(text, key=None):
    """Create holographic 3D button"""
    
    if st.button(text, key=key):
        return True
    
    # Add custom CSS for the button that was just created
    st.markdown(f"""
    <style>
    div.stButton > button:first-child {{
        background: linear-gradient(145deg, #2D1B69, #1A2332);
        color: white;
        border: 2px solid #00D4FF;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 
            0 0 20px rgba(0, 212, 255, 0.3),
            inset 5px 5px 10px rgba(255, 255, 255, 0.1),
            inset -5px -5px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }}
    
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 0 30px rgba(0, 212, 255, 0.5),
            inset 5px 5px 10px rgba(255, 255, 255, 0.2),
            inset -5px -5px 10px rgba(0, 0, 0, 0.4);
        border-color: #9D4EDD;
    }}
    
    div.stButton > button:active {{
        transform: translateY(0px);
        box-shadow: 
            0 0 15px rgba(0, 212, 255, 0.4),
            inset 3px 3px 8px rgba(0, 0, 0, 0.5);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return False

# ======================== CHART HELPERS ========================

def create_space_chart_theme():
    """Create consistent space-themed chart layout"""
    
    return dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Space Mono, monospace'),
        title=dict(font=dict(size=18, family='Orbitron, sans-serif')),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            color='white'
        )
    )

def apply_space_theme_to_fig(fig):
    """Apply space theme to plotly figure"""
    
    theme = create_space_chart_theme()
    fig.update_layout(**theme)
    return fig

# ======================== EXPORT FUNCTIONS ========================

__all__ = [
    'apply_custom_theme',
    'create_3d_earth',
    'create_space_trajectory',
    'calculate_orbital_parameters',
    'calculate_hohmann_transfer',
    'generate_space_weather_data',
    'generate_satellite_data',
    'generate_asteroid_data',
    'create_metric_card',
    'create_status_indicator',
    'create_loading_animation',
    'create_holographic_button',
    'create_space_chart_theme',
    'apply_space_theme_to_fig'
]
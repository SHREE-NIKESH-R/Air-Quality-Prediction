import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration - CLASSIC WHITE THEME
st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for CLASSIC WHITE THEME
st.markdown("""
    <style>
    /* Classic White Background */
    .main {
        background-color: #FFFFFF;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF;
    }
    
    /* Sidebar - Clean White */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E0E0E0;
    }
    
    /* ALL TEXT BLACK AND VISIBLE */
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #000000 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Input fields - Clear and Visible */
    [data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #CCCCCC !important;
        border-radius: 4px !important;
        padding: 8px !important;
    }
    
    [data-testid="stSidebar"] input:focus {
        border-color: #000000 !important;
        box-shadow: 0 0 0 1px #000000 !important;
    }
    
    /* Main content headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #000000;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Ensure all main content text is black */
    .stMarkdown {
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* Button styling - Classic Blue */
    .stButton>button {
        background-color: #1E88E5 !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stButton>button:hover {
        background-color: #1565C0 !important;
    }
    
    /* Table styling - Clean and Professional */
    .pollutant-table {
        width: 100%;
        background: white;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid #E0E0E0;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .pollutant-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .pollutant-table th {
        background: #F5F5F5;
        color: #000000;
        padding: 12px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #E0E0E0;
    }
    .pollutant-table td {
        padding: 12px;
        border-bottom: 1px solid #E0E0E0;
        color: #000000;
        font-weight: 400;
    }
    .pollutant-table tr:last-child td {
        border-bottom: none;
    }
    .pollutant-table tr:hover {
        background: #F8F8F8;
    }
    
    /* Info boxes styling */
    .stInfo {
        background-color: #E3F2FD !important;
        border: 1px solid #90CAF9 !important;
        color: #000000 !important;
    }
    
    .stSuccess {
        background-color: #E8F5E8 !important;
        border: 1px solid #81C784 !important;
        color: #000000 !important;
    }
    
    .stWarning {
        background-color: #FFF3E0 !important;
        border: 1px solid #FFB74D !important;
        color: #000000 !important;
    }
    
    .stError {
        background-color: #FFEBEE !important;
        border: 1px solid #E57373 !important;
        color: #000000 !important;
    }
    
    /* Make sure all text in info boxes is black */
    .stAlert * {
        color: #000000 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
    }
    
    /* Plotly chart text colors */
    .js-plotly-plot .plotly {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open('aqi_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run model training script first.")
        st.stop()

model, scaler = load_models()

# AQI category function
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "#4CAF50", "#E8F5E9", "Air quality is satisfactory"
    elif aqi <= 100:
        return "Moderate", "🟡", "#FFC107", "#FFF9C4", "Air quality is acceptable"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#FF9800", "#FFE0B2", "Sensitive groups may experience health effects"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#F44336", "#FFCDD2", "Everyone may experience health effects"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#9C27B0", "#E1BEE7", "Health alert: serious health effects"
    else:
        return "Hazardous", "🟤", "#795548", "#D7CCC8", "Health warnings of emergency conditions"

# Header
st.markdown('<p class="main-header">🌍 Air Quality Index Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict AQI based on pollutant concentrations (US EPA Standard)</p>', unsafe_allow_html=True)

# Sidebar for inputs - ONLY SLIDERS
st.sidebar.header("📊 Input Pollutant Values")


st.sidebar.markdown("**Adjust sliders to set pollutant values:**")

# PM2.5 Slider
pm25 = st.sidebar.slider(
    "PM2.5 (μg/m³)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.5,
    help="Fine particulate matter (diameter < 2.5 micrometers)"
)

# PM10 Slider
pm10 = st.sidebar.slider(
    "PM10 (μg/m³)",
    min_value=0.0,
    max_value=200.0,
    value=100.0,
    step=1.0,
    help="Coarse particulate matter (diameter < 10 micrometers)"
)

# NO2 Slider
no2 = st.sidebar.slider(
    "NO2 (μg/m³)",
    min_value=0.0,
    max_value=30.0,
    value=40.0,
    step=0.5,
    help="Nitrogen Dioxide"
)

# CO Slider
co = st.sidebar.slider(
    "CO (mg/m³)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.05,
    help="Carbon Monoxide"
)

# SO2 Slider
so2 = st.sidebar.slider(
    "SO2 (μg/m³)",
    min_value=0.0,
    max_value=60.0,
    value=15.0,
    step=0.5,
    help="Sulfur Dioxide"
)

# O3 Slider
o3 = st.sidebar.slider(
    "O3 (μg/m³)",
    min_value=0.0,
    max_value=40.0,
    value=20.0,
    step=0.5,
    help="Ozone"
)


predict_button = st.sidebar.button("🔮 Predict AQI", type="primary", use_container_width=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Current Input Values")
    
    # Custom HTML table
    table_html = f"""
    <div class="pollutant-table">
        <table>
            <tr>
                <th>Pollutant</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr><td>PM2.5</td><td>{pm25:.1f}</td><td>μg/m³</td></tr>
            <tr><td>PM10</td><td>{pm10:.1f}</td><td>μg/m³</td></tr>
            <tr><td>NO2</td><td>{no2:.1f}</td><td>μg/m³</td></tr>
            <tr><td>CO</td><td>{co:.1f}</td><td>mg/m³</td></tr>
            <tr><td>SO2</td><td>{so2:.1f}</td><td>μg/m³</td></tr>
            <tr><td>O3</td><td>{o3:.1f}</td><td>μg/m³</td></tr>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Pollutant visualization
    pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
    values = [pm25, pm10, no2, co, so2, o3]
    
    fig = go.Figure(data=[
        go.Bar(
            x=pollutants,
            y=values,
            marker_color=['#EF5350', '#26A69A', '#42A5F5', '#66BB6A', '#FFA726', '#AB47BC'],
            text=[f'{v:.1f}' for v in values],
            textposition='outside',
            textfont=dict(color='#000000', size=12, family='Arial')
        )
    ])
    
    fig.update_layout(
        title=dict(text="Pollutant Concentrations", font=dict(color='#000000', size=18, family='Arial')),
        xaxis_title=dict(text="Pollutant", font=dict(color='#000000', family='Arial')),
        yaxis_title=dict(text="Concentration", font=dict(color='#000000', family='Arial')),
        height=350,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(gridcolor='#E0E0E0', color='#000000'),
        yaxis=dict(gridcolor='#E0E0E0', color='#000000'),
        font=dict(color='#000000', family='Arial'),
        margin=dict(t=60, b=40, l=50, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Result")
    
    if predict_button:
        # Prepare input for prediction
        input_data = np.array([[pm25, pm10, no2, co, so2, o3]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = prediction + 4
        prediction = max(0, prediction)
        
        # Get AQI category
        category, emoji, color, bg_color, description = get_aqi_category(prediction)
        
        # Display prediction with clean design
        st.markdown(f"""
            <div style='background: {bg_color};
                        padding: 2rem; 
                        border-radius: 8px; 
                        text-align: center;
                        border: 2px solid {color};
                        margin-bottom: 1rem;'>
                <h1 style='color: {color}; margin: 0; font-size: 3rem; font-weight: bold;'>
                    {emoji} {prediction:.0f}
                </h1>
                <h2 style='color: {color}; margin-top: 0.5rem; font-weight: 600;'>
                    {category}
                </h2>
                <p style='color: #000000; font-size: 1.1rem; margin-top: 0.5rem;'>
                    {description}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Health recommendations
        st.markdown("### 📋 Health Recommendations")
        
        if prediction <= 50:
            st.success("✅ **Good:** Air quality is satisfactory. Enjoy outdoor activities!")
        elif prediction <= 100:
            st.info("ℹ️ **Moderate:** Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exertion.")
        elif prediction <= 150:
            st.warning("⚠️ **Unhealthy for Sensitive Groups:** Children, elderly, and people with respiratory conditions should reduce outdoor activities.")
        elif prediction <= 200:
            st.warning("⚠️ **Unhealthy:** Everyone should reduce prolonged outdoor exertion. Sensitive groups should avoid outdoor activities.")
        elif prediction <= 300:
            st.error("🚨 **Very Unhealthy:** Avoid prolonged outdoor exertion. Everyone should stay indoors when possible.")
        else:
            st.error("🚨 **Hazardous:** Avoid all outdoor activities. Use air purifiers indoors and keep windows closed.")
    
    else:
        st.info("👈 **Adjust the sliders in the sidebar and click 'Predict AQI' to see results**")
        
        # AQI scale reference
        st.markdown("### 📊 AQI Scale Reference")
        
        scale_html = """
        <div class="pollutant-table">
            <table>
                <tr>
                    <th>AQI Range</th>
                    <th>Category</th>
                    <th>Color Code</th>
                </tr>
                <tr><td>0 - 50</td><td>Good</td><td>🟢 Green</td></tr>
                <tr><td>51 - 100</td><td>Moderate</td><td>🟡 Yellow</td></tr>
                <tr><td>101 - 150</td><td>Unhealthy for Sensitive Groups</td><td>🟠 Orange</td></tr>
                <tr><td>151 - 200</td><td>Unhealthy</td><td>🔴 Red</td></tr>
                <tr><td>201 - 300</td><td>Very Unhealthy</td><td>🟣 Purple</td></tr>
                <tr><td>301 - 500</td><td>Hazardous</td><td>🟤 Maroon</td></tr>
            </table>
        </div>
        """
        st.markdown(scale_html, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #000000; padding: 1rem; background-color: #F5F5F5; border-radius: 4px; border: 1px solid #E0E0E0;'>
        <p style='margin: 0.5rem 0; font-weight: 500;'>📍 Based on US EPA Air Quality Standards</p>
        <p style='margin: 0.5rem 0;'>🔬 Powered by Machine Learning | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)
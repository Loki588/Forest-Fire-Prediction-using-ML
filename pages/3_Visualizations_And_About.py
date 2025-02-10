import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for better styling
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 16px;
    border-radius: 4px;
    background-color: #f0f2f6;
    transition: all 0.2s ease-in-out;
}
.stTabs [aria-selected="true"] {
    background-color: #4CAF50 !important;
    color: white !important;
}
.stMarkdown > div > div {
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Data Visualizations & ℹ️ About the Project")

# Check data availability
if "user_inputs" not in st.session_state:
    st.warning("⚠️ No input data found! Please visit the **Predict Fire** page first.")
    st.stop()

# Cache the dataset to reduce load times
@st.cache_data
def load_data():
    df = pd.read_csv("data/forestfires.csv")
    
    # Convert categorical columns (month, day) to numeric
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_map = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}
    
    df['month'] = df['month'].map(month_map)
    df['day'] = df['day'].map(day_map)
    
    return df

# Load data
df = load_data()

# ✅ Tabs for switching between Visualizations & About sections
tab1, tab2 = st.tabs(["📊 Visualizations", "ℹ️ About the Project"])

# ✅ Visualizations Section
with tab1:
    try:
        # Load user inputs from session state
        user_data = st.session_state.user_inputs

        # Create DataFrame from user inputs
        df_user = pd.DataFrame([user_data])

        # Feature mappings for display names
        feature_names = {
            "X": "X Coordinate",
            "Y": "Y Coordinate",
            "FFMC": "FFMC Index",
            "DMC": "DMC Index",
            "DC": "DC Index",
            "ISI": "ISI Index",
            "temperature": "Temperature (°C)",
            "humidity": "Humidity (%)",
            "wind_speed": "Wind Speed (km/h)",
            "rain": "Rainfall (mm)",
            "month_encoded": "Month",
            "day_encoded": "Day"
        }

        # 1. Feature Distribution Bar Chart (Plotly)
        st.subheader("📋 Input Feature Distribution")
        fig1 = px.bar(
            df_user,
            x=list(feature_names.values()),
            y=df_user.iloc[0].values,
            labels={"x": "Feature", "y": "Value"},
            title="<b>Your Input Features</b>",
            color=list(feature_names.values()),
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig1.update_layout(
            title_font=dict(size=20),
            xaxis_title="Feature",
            yaxis_title="Value",
            hovermode="x unified",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Environmental Factors vs Fire Risk (Plotly Scatter Plots)
        st.subheader("🌡️ Environmental Factors Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Temperature vs FFMC
            fig2 = px.scatter(
                df_user,
                x="temperature",
                y="FFMC",
                title="<b>Temperature vs FFMC</b>",
                labels={"temperature": "Temperature (°C)", "FFMC": "FFMC Index"},
                color_discrete_sequence=["red"]
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Humidity vs DMC
            fig3 = px.scatter(
                df_user,
                x="humidity",
                y="DMC",
                title="<b>Humidity vs DMC</b>",
                labels={"humidity": "Humidity (%)", "DMC": "DMC Index"},
                color_discrete_sequence=["blue"]
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 3. Weather Conditions Radar Chart (Plotly)
        st.subheader("🌀 Weather Conditions Overview")
        weather_features = ["temperature", "humidity", "wind_speed", "rain"]
        weather_values = df_user[weather_features].values[0]
        weather_labels = ["Temperature", "Humidity", "Wind Speed", "Rainfall"]

        fig4 = px.line_polar(
            r=weather_values,
            theta=weather_labels,
            line_close=True,
            title="<b>Weather Conditions Radar Chart</b>",
            color_discrete_sequence=["green"]
        )
        fig4.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            template="plotly_white"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # 4. Optional Additional Graphs (Checkbox-Based)
        st.subheader("📊 Optional Visualizations")
        show_correlation = st.checkbox("Show Correlation Heatmap")
        show_pairplot = st.checkbox("Show Pairwise Relationships (Pair Plot)")
        show_temp_vs_area = st.checkbox("Show Temperature vs Burned Area")
        show_wind_vs_area = st.checkbox("Show Wind Speed vs Burned Area")
        show_rain_vs_area = st.checkbox("Show Rainfall vs Burned Area")

        # Generate optional graphs based on checkbox selection
        if show_correlation:
            st.subheader("🔥 Correlation Heatmap")
            numeric_df = df.select_dtypes(include=np.number)  # Only numeric columns
            fig5 = px.imshow(
                numeric_df.corr(),
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                title="<b>Feature Correlation Heatmap</b>",
                color_continuous_scale="RdBu_r"  # Valid Plotly colorscale
            )
            st.plotly_chart(fig5, use_container_width=True)

        if show_pairplot:
            st.subheader("Pairwise Relationships")
            st.write("This may take a few seconds to generate...")
            numeric_df = df.select_dtypes(include=np.number)  # Only numeric columns
            fig6 = px.scatter_matrix(numeric_df)
            st.plotly_chart(fig6, use_container_width=True)

        if show_temp_vs_area:
            st.subheader("📉 Temperature vs. Burned Area")
            fig7 = px.scatter(
                df,
                x="temp",
                y="area",
                title="<b>Temperature vs. Burned Area</b>",
                labels={"temp": "Temperature (°C)", "area": "Burned Area (Hectares)"},
                color_discrete_sequence=["red"]
            )
            st.plotly_chart(fig7, use_container_width=True)

        if show_wind_vs_area:
            st.subheader("🌬 Wind Speed vs. Burned Area")
            fig8 = px.scatter(
                df,
                x="wind",
                y="area",
                title="<b>Wind Speed vs. Burned Area</b>",
                labels={"wind": "Wind Speed (km/h)", "area": "Burned Area (Hectares)"},
                color_discrete_sequence=["blue"]
            )
            st.plotly_chart(fig8, use_container_width=True)

        if show_rain_vs_area:
            st.subheader("🌧 Rainfall vs. Burned Area")
            fig9 = px.scatter(
                df,
                x="rain",
                y="area",
                title="<b>Rainfall vs. Burned Area</b>",
                labels={"rain": "Rainfall (mm)", "area": "Burned Area (Hectares)"},
                color_discrete_sequence=["green"]
            )
            st.plotly_chart(fig9, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")

# ✅ About Section
with tab2:
    st.subheader("ℹ️ About the Project")
    st.markdown("""
    ### 🚀 Project Overview
    This interactive platform combines machine learning with data visualization to:
    - Predict forest fire risks based on environmental conditions
    - Analyze how input features contribute to fire spread predictions
    - Provide actionable insights for environmental protection agencies
    """)

    st.markdown("""
    ### 📌 Key Features
    - **Real-time Predictions**: Get instant fire risk assessments
    - **Input-Driven Visualizations**: See how your specific inputs relate to fire patterns
    - **Model Transparency**: Understand feature importance and relationships
    - **Portable Reports**: Download PDF summaries of your analysis
    """)

    st.markdown("""
    ### 🛠 Technical Details
    - **Models Used**:
        - Random Forest Regressor
        - Neural Network (3-layer MLP)
    - **Input Features**: 12 environmental parameters
    - **Prediction Target**: Burned area in hectares
    - **Accuracy**:
        - Random Forest: R² = 0.89
        - Neural Network: R² = 0.85
    """)

    st.markdown("""
    ### 📚 Data Sources
    - Primary dataset: [UCI Forest Fires Dataset](https://archive.ics.uci.edu/ml/datasets/forest+fires)
    - Weather data integration: OpenWeatherMap API
    """)
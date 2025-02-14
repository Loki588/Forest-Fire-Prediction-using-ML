import streamlit as st
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import math
import folium
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from config import OPENWEATHER_API_KEY, FIRMS_API_KEY

# Load the trained neural network model
@st.cache_resource
def load_nn_model():
    try:
        custom_objects = {"mse": MeanSquaredError()}
        model = tf.keras.models.load_model("forest_fire_nn_model.h5", custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading neural network model: {str(e)}")
        return None

model = load_nn_model()

# API Keys


# Function to get city coordinates
def get_city_coordinates(city_name):
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        return None, None
    except Exception as e:
        st.error(f"Error fetching city coordinates: {str(e)}")
        return None, None

# Function to fetch weather data
def fetch_weather_data(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "rainfall": data.get("rain", {}).get("1h", 0.0)
        }
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

# Function to fetch fire data
def fetch_fire_data(lat, lon):
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_API_KEY}/VIIRS_SNPP_NRT/world/10"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        fire_data = response.text
        fire_count = fire_data.count(f"{lat:.2f},{lon:.2f}")
        return fire_count
    except Exception as e:
        st.error(f"Error fetching fire data: {str(e)}")
        return None

# Function to calculate FFMC (Fine Fuel Moisture Code)
def calculate_ffmc(temp, rh, wind, rain, previous_ffmc):
    if previous_ffmc is None:
        previous_ffmc = 85.0  # Default initial value
    mo = (147.2 * (101 - previous_ffmc)) / (59.5 + previous_ffmc)
    if rain > 0.5:
        rf = rain - 0.5
        mo = mo + 42.5 * rf * math.exp(-100 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
    ed = 0.942 * (rh ** 0.679) + 11 * math.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * rh))
    ew = 0.618 * (rh ** 0.753) + 10 * math.exp((rh - 100) / 10) + 0.18 * (21.1 - temp) * (1 - math.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * math.sqrt(wind) * (1 - (rh / 100) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        mo = ed + (mo - ed) * 10 ** (-kd)
    if mo < ew:
        kl = 0.424 * (1 - ((100 - rh) / 100) ** 1.7) + 0.0694 * math.sqrt(wind) * (1 - ((100 - rh) / 100) ** 8)
        kw = kl * 0.581 * math.exp(0.0365 * temp)
        mo = ew - (ew - mo) * 10 ** (-kw)
    ffmc = (59.5 * (250 - mo)) / (147.2 + mo)
    return ffmc

# Function to calculate DMC (Duff Moisture Code)
def calculate_dmc(temp, rh, rain, previous_dmc):
    if previous_dmc is None:
        previous_dmc = 6.0  # Default initial value
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20 + math.exp(5.6348 - previous_dmc / 43.43)
        b = 100 / (0.5 + 0.3 * previous_dmc)
        if previous_dmc <= 33:
            b = 14 - 1.3 * math.log(previous_dmc)
        elif previous_dmc > 65:
            b = 6.2 * math.log(previous_dmc) - 17.2
        mo = mo + 1000 * re / (48.77 + b * re)
        dmc = 43.43 * (5.6348 - math.log(mo - 20))
    else:
        dmc = previous_dmc
    return dmc

# Function to calculate DC (Drought Code)
def calculate_dc(temp, rain, previous_dc):
    if previous_dc is None:
        previous_dc = 15.0  # Default initial value
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800 * math.exp(-previous_dc / 400)
        qo = qo + 3.937 * rd
        dc = 400 * math.log(800 / qo)
    else:
        dc = previous_dc
    return dc

# Function to calculate ISI (Initial Spread Index)
def calculate_isi(ffmc, wind_speed):
    mo = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    ff = 19.115 * math.exp(mo * -0.1386) * (1 + (mo ** 5.31) / 49300000)
    isi = ff * math.exp(0.05039 * wind_speed)
    return isi

# Main application
def city_weather_page():
    st.title("🌍 City-Based Forest Fire Risk Assessment")

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        city_name = st.text_input("🏙️ Enter City Name", "London")

    # Automatically set the current month and day
    current_date = datetime.now()
    current_month = current_date.strftime("%b").lower()  # e.g., "feb" for February
    current_day = current_date.strftime("%a").lower()    # e.g., "fri" for Friday

    # Debug: Print current month and day (optional)
    st.write(f"Current Month: {current_month}, Current Day: {current_day}")

    predict_button = st.button("🚀 Predict Fire Risk")

    # Reset session state if inputs change
    if "previous_city" not in st.session_state:
        st.session_state.previous_city = city_name

    if city_name != st.session_state.previous_city:
        # Clear session state if inputs change
        st.session_state.clear()
        st.session_state.previous_city = city_name

    if predict_button:
        with st.spinner("Fetching data and calculating risks..."):
            # Get city coordinates
            lat, lon = get_city_coordinates(city_name)
            if not lat or not lon:
                st.error("Could not find coordinates for this city")
                st.stop()

            # Fetch weather data
            weather_data = fetch_weather_data(lat, lon)
            if not weather_data:
                st.error("Failed to fetch weather data")
                st.stop()

            # Convert coordinates to grid
            X = int((lon + 180) / 36) % 10
            Y = int((lat + 90) / 18) % 10

            # Calculate fire indices
            ffmc = calculate_ffmc(
                weather_data["temperature"],
                weather_data["humidity"],
                weather_data["wind_speed"],
                weather_data["rainfall"],
                None  # Default value for previous_ffmc
            )
            dmc = calculate_dmc(
                weather_data["temperature"],
                weather_data["humidity"],
                weather_data["rainfall"],
                None  # Default value for previous_dmc
            )
            dc = calculate_dc(
                weather_data["temperature"],
                weather_data["rainfall"],
                None  # Default value for previous_dc
            )
            isi = calculate_isi(ffmc, weather_data["wind_speed"])

            # Prepare features with current month and day
            features = np.array([[
                X, Y,
                ffmc, dmc, dc, isi,
                weather_data["temperature"],
                weather_data["humidity"],
                weather_data["wind_speed"],
                weather_data["rainfall"],
                ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].index(current_month),
                ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'].index(current_day)
            ]])

            # Make prediction
            try:
                prediction = model.predict(features)[0][0]
                risk_level = "🔥 HIGH RISK" if prediction > 0.5 else "✅ LOW RISK"

                # Display results
                st.subheader("Analysis Results")
                st.write("### Weather Data")
                cols = st.columns(4)
                cols[0].metric("🌡 Temperature", f"{weather_data['temperature']} °C")
                cols[1].metric("💧 Humidity", f"{weather_data['humidity']}%")
                cols[2].metric("🌬 Wind Speed", f"{weather_data['wind_speed']} km/h")
                cols[3].metric("🌧 Rainfall", f"{weather_data['rainfall']} mm")

                # Display fire indices
                st.write("### Fire Weather Indices")
                cols = st.columns(4)
                cols[0].metric("FFMC Index", f"{ffmc:.2f}")
                cols[1].metric("DMC Index", f"{dmc:.2f}")
                cols[2].metric("DC Index", f"{dc:.2f}")
                cols[3].metric("ISI Index", f"{isi:.2f}")

                # Display prediction result
                st.divider()
                st.subheader(f"Prediction Result: {risk_level}")
                progress_value = float(np.clip(prediction, 0, 1))
                st.progress(progress_value)
                st.write(f"Predicted fire spread probability: {prediction:.2%}")


                # Display the city's location and risk level on an interactive map
                st.subheader("📍 City Location and Risk Level")
                m = folium.Map(location=[lat, lon], zoom_start=10)
                folium.Marker(
                location=[lat, lon],
                popup=f"{city_name}: {risk_level}",
                icon=folium.Icon(color="red" if risk_level == "🔥 HIGH RISK" else "green")
                ).add_to(m)
                folium_static(m)


                # Explain how weather data and fire indices affect risk
                st.subheader("📊 How Weather Data and Fire Indices Affect Risk")
                st.write("The fire risk prediction is influenced by the following factors:")


                # Create a DataFrame for visualization
                factors = {
                    "Factor": ["Temperature", "Humidity", "Wind Speed", "Rainfall", "FFMC", "DMC", "DC", "ISI"],
                    "Value": [
                        weather_data["temperature"],
                        weather_data["humidity"],
                        weather_data["wind_speed"],
                        weather_data["rainfall"],
                        ffmc, dmc, dc, isi
                        ],
                    "Impact": [
                        "Higher temperatures increase fire risk.",
                        "Lower humidity increases fire risk.",
                        "Higher wind speeds increase fire spread.",
                        "Rainfall reduces fire risk.",
                        "Higher FFMC indicates drier fuel, increasing fire risk.",
                        "Higher DMC indicates drier deep fuel, increasing fire risk.",
                        "Higher DC indicates drought conditions, increasing fire risk.",
                        "Higher ISI indicates faster fire spread."
                        ]
                        }
                df_factors = pd.DataFrame(factors)

                # Display the DataFrame with styled formatting
                st.dataframe(
                    df_factors,
                    column_config={
                        "Factor": "Factor",
                        "Value": st.column_config.NumberColumn("Value", format="%.2f"),
                        "Impact": "Impact"
                    },
                    hide_index=True,
                    use_container_width=True
                )

                        # ----------------------------------------
                # Modern Visualization 1: Interactive Bar Chart (Plotly)
                # ----------------------------------------
                st.subheader("📊 Impact of Factors on Fire Risk")
                fig1 = px.bar(
                    df_factors,
                    x="Factor",
                    y="Value",
                    color="Factor",
                    text="Value",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title="<b>Factor Contributions to Fire Risk</b>",
                    template="plotly_white"
                )
                fig1.update_layout(
                    title_font=dict(size=20),
                    xaxis_title="Factor",
                    yaxis_title="Value",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig1, use_container_width=True)


                # ----------------------------------------
                # Modern Visualization 2: Radar Chart (Plotly)
                # ----------------------------------------
                st.subheader("🌪️ Weather Conditions Radar Chart")
                fig2 = px.line_polar(
                    df_factors[df_factors["Factor"].isin(["Temperature", "Humidity", "Wind Speed", "Rainfall"])],
                    r="Value",
                    theta="Factor",
                    line_close=True,
                    color_discrete_sequence=["#FF4B4B"],
                    title="<b>Weather Profile</b>"
                )
                fig2.update_layout(polar=dict(radialaxis=dict(visible=True)))
                st.plotly_chart(fig2, use_container_width=True)


                # ----------------------------------------
                # Modern Visualization 3: Fire Indices Relationship (Plotly Scatter)
                # ----------------------------------------
                st.subheader("🔥 Fire Indices vs. Risk")
                fig3 = px.scatter(
                    df_factors[4:8],  # FFMC, DMC, DC, ISI
                    x="Value",
                    y="Factor",
                    color="Factor",
                    size="Value",
                    labels={"Value": "Index Value"},
                    title="<b>Fire Index Severity</b>"
                )
                st.plotly_chart(fig3, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")




if __name__ == "__main__":
    city_weather_page()
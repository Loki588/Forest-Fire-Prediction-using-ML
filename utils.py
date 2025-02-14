import streamlit as st 
import math
import requests

OPENWEATHER_API_KEY = "0c18a1583614fbf74003bc19f6c89a72"

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

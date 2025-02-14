import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from streamlit.components.v1 import html
from config import GOOGLE_MAPS_API_KEY, NASA_FIRMS_API_KEY

# API Keys (Replace with your own)


# ------------------------- Helper Functions ------------------------- #
@st.cache_data(ttl=3600)
def get_fire_data(main_url, map_key, source, area, day_range, date):
    url = f"{main_url}/{map_key}/{source}/{area}/{day_range}/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(url)
    else:
        st.error("Failed to fetch data")
        return pd.DataFrame()

def fetch_data_parallel(main_url, map_key, source, area_coords, start_date, end_date):
    fire_data = pd.DataFrame()
    current_date = start_date
    data_found = False
    
    with ThreadPoolExecutor() as executor:
        futures = []
        while current_date <= end_date:
            chunk_end_date = min(current_date + timedelta(days=9), end_date)
            futures.append(
                executor.submit(
                    get_fire_data, main_url, map_key, source, area_coords, 10,
                    current_date.strftime('%Y-%m-%d')
                )
            )
            current_date = chunk_end_date + timedelta(days=1)
        
        for future in futures:
            data = future.result()
            if not data.empty:
                fire_data = pd.concat([fire_data, data])
                data_found = True
    return fire_data, data_found

def classify_severity(bright_ti5):
    if bright_ti5 < 320:
        return "Low"
    elif 320 <= bright_ti5 < 350:
        return "Medium"
    else:
        return "High"

# ------------------------- Google Maps Integration ------------------------- #
def google_maps_click_handler():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fire Area Selection</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={GOOGLE_MAPS_API_KEY}&loading=async&callback=initMap"></script>
        <script>
            let map;
            let marker;
            
            function initMap() {{
                map = new google.maps.Map(document.getElementById('map'), {{
                    center: {{lat: 0, lng: 0}},
                    zoom: 2,
                    gestureHandling: "cooperative"
                }});
                
                map.addListener('click', (e) => {{
                    const lat = e.latLng.lat();
                    const lng = e.latLng.lng();
                    
                    if (marker) {{
                        marker.setMap(null);
                    }}
                    
                    marker = new google.maps.Marker({{
                        position: e.latLng,
                        map: map,
                        title: "Selected Location"
                    }});
                    
                    // Display coordinates below the map
                    document.getElementById('coordinates').innerHTML = 
                        `<p>Selected Coordinates: <strong>${{lat.toFixed(4)}}, ${{lng.toFixed(4)}}</strong></p>`;
                }});
            }}
        </script>
        <style>
            #map {{
                height: 400px;
                width: 100%;
            }}
            #coordinates {{
                margin-top: 10px;
                font-size: 16px;
            }}
        </style>
    </head>
    <body style="margin:0;padding:0;">
        <div id="map"></div>
        <div id="coordinates"></div>
    </body>
    </html>
    """

# ------------------------- Main Application ------------------------- #
def main():
    st.title("Historical Fire Data Analysis")
    
    # Initialize session state
    if "fire_data" not in st.session_state:
        st.session_state.fire_data = None

    # ------------------------- Sidebar Controls ------------------------- #
    st.sidebar.header("Data Selection Parameters")
    
    # Google Maps coordinate selection
    st.sidebar.markdown("### Step 1: Select Area")
    html(google_maps_click_handler(), height=450)
    
    # Manual coordinate entry
    st.sidebar.markdown("### Step 2: Enter Coordinates")
    lat = st.sidebar.number_input("Latitude", value=0.0, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=0.0, format="%.4f")
    
    # Date range selection
    st.sidebar.markdown("### Step 3: Select Date Range")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    
    # ------------------------- Data Processing ------------------------- #
    if st.sidebar.button("Analyze Fire History"):
        if lat == 0.0 and lon == 0.0:
            st.error("Please select a location on the map and enter coordinates!")
            return
            
        with st.spinner("Fetching and processing fire data..."):
            # Define area coordinates
            xmin = lon - 1
            xmax = lon + 1
            ymin = lat - 1
            ymax = lat + 1
            area_coords = f"{xmin},{ymin},{xmax},{ymax}"
            
            # Fetch data
            main_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            source = "VIIRS_SNPP_NRT"
            fire_data, data_found = fetch_data_parallel(main_url, NASA_FIRMS_API_KEY, 
                                                      source, area_coords, start_date, end_date)
            
            if not data_found:
                st.error("No fire data found for selected area and date range")
                return
                
            # Process data
            fire_data['acq_date'] = pd.to_datetime(fire_data['acq_date'])
            fire_data['year'] = fire_data['acq_date'].dt.year
            fire_data['month'] = fire_data['acq_date'].dt.month
            fire_data['severity'] = fire_data['bright_ti5'].apply(classify_severity)
            st.session_state.fire_data = fire_data
    
    # ------------------------- Visualizations ------------------------- #
    if st.session_state.fire_data is not None:
        fire_data = st.session_state.fire_data
        
        st.markdown("## Fire Event Analysis")
        
        # Folium Map Visualization
        st.markdown("### Fire Events Map")
        m = folium.Map(location=[lat, lon], zoom_start=7)
        
        for _, row in fire_data.iterrows():
            color = "green" if row['severity'] == "Low" else "orange" if row['severity'] == "Medium" else "red"
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Date: {row['acq_date'].date()}<br>Brightness: {row['bright_ti5'] - 273.15:.2f}°C<br>Severity: {row['severity']}"
            ).add_to(m)
        
        folium_static(m)

        # ------------------------- Fire Activity Trends ------------------------- #
        st.markdown("## Fire Activity Trends")

        # Most Active Months
        st.markdown("### Most Active Months")
        monthly_counts = fire_data.groupby('month').size().reset_index(name='count')
        monthly_counts['month'] = monthly_counts['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        top_months = monthly_counts.nlargest(3, 'count')
        st.write(f"Top 3 months with the highest fire activity: {', '.join(top_months['month'])}")

        # Monthly Fire Activity Chart
        st.markdown("### Monthly Fire Activity")
        fig = px.bar(
            monthly_counts,
            x='month',
            y='count',
            labels={'count': 'Number of Fire Events', 'month': 'Month'},
            title="Monthly Fire Activity"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------- Most Active Regions ------------------------- #
        st.markdown("### Most Active Regions")
        
        # Group by rounded coordinates (2 decimal places ≈ 1.1 km² area)
        fire_data['lat_rounded'] = fire_data['latitude'].round(2)
        fire_data['lon_rounded'] = fire_data['longitude'].round(2)
        
        region_counts = fire_data.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
        top_regions = region_counts.nlargest(3, 'count')
        
        # Display as expandable sections
        for idx, row in top_regions.iterrows():
            with st.expander(f"Region {idx+1}: {row['count']} fires"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latitude", f"{row['lat_rounded']:.2f}")
                with col2:
                    st.metric("Longitude", f"{row['lon_rounded']:.2f}")
                st.write(f"**Total Fires:** {row['count']}")
                
                # Show on mini map
                mini_map = folium.Map(location=[row['lat_rounded'], row['lon_rounded']], zoom_start=8)
                folium.Marker(
                    [row['lat_rounded'], row['lon_rounded']],
                    tooltip="Hotspot Area"
                ).add_to(mini_map)
                folium_static(mini_map, width=400)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

# Hardcoded NASA FIRMS API Key
NASA_FIRMS_API_KEY = "2402ceb75d735d32221a262c701f504e"

# Function to fetch fire data
@st.cache_data(ttl=3600)  # Cache data for 1 hour to reduce redundant API calls
def get_fire_data(main_url, map_key, source, area, day_range, date):
    url = f"{main_url}/{map_key}/{source}/{area}/{day_range}/{date}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(url)
    else:
        st.error("Failed to fetch data")
        return pd.DataFrame()

# Function to fetch data in parallel for multiple date ranges
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
                    get_fire_data, main_url, map_key, source, area_coords, 10, current_date.strftime('%Y-%m-%d')
                )
            )
            current_date = chunk_end_date + timedelta(days=1)

        for future in futures:
            data = future.result()
            if not data.empty:
                fire_data = pd.concat([fire_data, data])
                data_found = True

    return fire_data, data_found

# Function to classify fire severity based on brightness temperature
def classify_severity(bright_ti5):
    if bright_ti5 < 320:
        return "Low"
    elif 320 <= bright_ti5 < 350:
        return "Medium"
    else:
        return "High"

# Streamlit app
def main():
    st.title("Historical Fire Data Analysis")

    # Initialize session state for fire data
    if "fire_data" not in st.session_state:
        st.session_state.fire_data = None

    # Sidebar for user inputs
    st.sidebar.header("User Input Parameters")

    # Clearer instructions for users
    st.sidebar.markdown("""
    **Instructions:**

    1. **Select Area on the Map**: Click on the map to choose a location or manually enter latitude and longitude.

    2. **Select Date Range**: Choose a start and end date to fetch fire data.

    3. **Start Processing**: Click the button to fetch and visualize fire data.
    """)

    # Map for selecting coordinates
    st.sidebar.write("Select Area on the Map:")
    m = folium.Map(location=[0, 0], zoom_start=2)
    m.add_child(folium.LatLngPopup())  # Allows clicking on the map to get coordinates

    # Use session state to store coordinates
    if "lat" not in st.session_state:
        st.session_state.lat = 0.0
    if "lon" not in st.session_state:
        st.session_state.lon = 0.0

    # Handle map click events
    map_data = folium_static(m)
    if map_data:
        if "last_clicked" in st.session_state:
            st.session_state.lat = st.session_state.last_clicked["lat"]
            st.session_state.lon = st.session_state.last_clicked["lng"]

    # Get coordinates from the map
    st.sidebar.write("Selected Coordinates:")
    lat = st.sidebar.number_input("Latitude", value=st.session_state.lat, min_value=-90.0, max_value=90.0)
    lon = st.sidebar.number_input("Longitude", value=st.session_state.lon, min_value=-180.0, max_value=180.0)

    # Validate coordinates
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        st.sidebar.error("Invalid coordinates. Latitude must be between -90 and 90, and longitude between -180 and 180.")
        return

    # Define area coordinates
    xmin = lon - 1  # Adjust the bounding box as needed
    xmax = lon + 1
    ymin = lat - 1
    ymax = lat + 1
    area_coords = f"{xmin},{ymin},{xmax},{ymax}"

    # Date range selection
    st.sidebar.write("Select Date Range:")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    # Button to start processing
    if st.sidebar.button("Start Processing"):
        # Clear previous fire data from session state
        st.session_state.fire_data = None

        with st.spinner("Fetching data and calculating risks..."):
            # Fetch fire data for the selected date range
            main_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            source = "VIIRS_SNPP_NRT"

            # Fetch data in parallel
            fire_data, data_found = fetch_data_parallel(main_url, NASA_FIRMS_API_KEY, source, area_coords, start_date, end_date)

            if not data_found:
                st.error("No fire data found for the selected area and date range. Please select a different area.")
                return

            if not fire_data.empty:
                # Convert acquisition date to datetime
                fire_data['acq_date'] = pd.to_datetime(fire_data['acq_date'])

                # Add a year and month column for grouping
                fire_data['year'] = fire_data['acq_date'].dt.year
                fire_data['month'] = fire_data['acq_date'].dt.month

                # Classify fire severity
                fire_data['severity'] = fire_data['bright_ti5'].apply(classify_severity)

                # Store fire data in session state
                st.session_state.fire_data = fire_data

                # Debug: Print fetched data years
                st.write(f"Fetched data years: {fire_data['year'].unique()}")

    # Check if fire data is available in session state
    if st.session_state.fire_data is not None:
        fire_data = st.session_state.fire_data

        # Create a time-slider for animation
        st.write("### Fire Events Over Time")
        year_range = st.slider(
            "Select Year",
            min_value=int(fire_data['year'].min()),
            max_value=int(fire_data['year'].max()),
            value=int(fire_data['year'].min()),
            step=1
        )

        # Filter data for the selected year
        filtered_data = fire_data[fire_data['year'] == year_range]

        # Create a Folium map
        m = folium.Map(location=[lat, lon], zoom_start=7)

        # Add points to the map for the selected year with severity-based colors
        for idx, row in filtered_data.iterrows():
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

        # Display the map in Streamlit
        folium_static(m)

        # Add a slider for animation speed
        st.write("### Adjust Animation Speed")
        animation_speed = st.select_slider(
            "Select Animation Speed",
            options=["Slow", "Medium", "Fast"],
            value="Medium"
        )

        # Map the selected speed to a frame duration in milliseconds
        speed_mapping = {
            "Slow": 3000,  # 3 seconds per frame
            "Medium": 1500,  # 1.5 seconds per frame
            "Fast": 800  # 0.8 seconds per frame
        }

        frame_duration = speed_mapping[animation_speed]

        # Optional: Display fire events over time using Plotly
        st.write("### Fire Events Over Time (Plotly Animation)")
        fire_data['date'] = fire_data['acq_date'].dt.date
        fig = px.scatter_geo(
            fire_data,
            lat='latitude',
            lon='longitude',
            animation_frame='year',  # Animate by year
            color='severity',
            hover_name='acq_date',
            scope='world',
            title="Fire Events Over Time",
            labels={'severity': 'Fire Severity', 'acq_date': 'Date'},
            projection="natural earth"  # Use a more detailed projection
        )

        # Add country borders and city names
        fig.update_geos(
            showcountries=True,  # Show country borders
            countrycolor="Black",  # Color of country borders
            showland=True,  # Show land masses
            landcolor="lightgray",  # Color of land masses
            showocean=True,  # Show oceans
            oceancolor="lightblue",  # Color of oceans
            showlakes=True,  # Show lakes
            lakecolor="blue"  # Color of lakes
        )

        # Adjust animation speed
        fig.update_layout(
            geo=dict(projection_scale=10),  # Adjust map zoom
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

        st.plotly_chart(fig)

        # Summarize trends in the data
        st.write("### Fire Activity Trends")

        # Most Active Months
        st.write("#### Most Active Months")
        monthly_counts = fire_data.groupby('month').size().reset_index(name='count')
        monthly_counts['month'] = monthly_counts['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        })
        top_months = monthly_counts.nlargest(3, 'count')
        st.write(f"Top 3 months with the highest fire activity: {', '.join(top_months['month'])}")

        # Bar Chart for Monthly Trends
        st.write("#### Monthly Fire Activity")
        fig_monthly = px.bar(
            monthly_counts,
            x='month',
            y='count',
            labels={'count': 'Number of Fire Events', 'month': 'Month'},
            title="Monthly Fire Activity"
        )
        st.plotly_chart(fig_monthly)

        # Most Active Regions
        st.write("#### Most Active Regions")
        region_counts = fire_data.groupby(['latitude', 'longitude']).size().reset_index(name='count')
        top_regions = region_counts.nlargest(3, 'count')
        st.write("Top 3 regions with the highest fire activity:")
        for idx, row in top_regions.iterrows():
            st.write(f"- Latitude: {row['latitude']}, Longitude: {row['longitude']} (Fires: {row['count']})")

if __name__ == "__main__":
    main()
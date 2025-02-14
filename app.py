import streamlit as st

st.set_page_config(
    page_title="Forest Fire Prediction",
    layout="wide",
    initial_sidebar_state="expanded" )
# Sidebar with logo (fixed)
with st.sidebar:
    st.image("forest logo.png", use_container_width=True)  # Replaced st.logo with st.image
 

import sqlite3
import hashlib
import numpy as np
import joblib
import tensorflow as tf
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from tensorflow.keras.losses import MeanSquaredError
from fpdf import FPDF
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from utils import( 
    get_city_coordinates,
    fetch_weather_data,
    fetch_fire_data,
    calculate_ffmc, 
    calculate_dmc, 
    calculate_dc, 
    calculate_isi
)

from config import OPENWEATHER_API_KEY,FIRMS_API_KEY,MAILGUN_API_KEY,MAILGUN_DOMAIN


# Database Configuration
DATABASE_NAME = "fire_prediction.db"

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
                  home_city TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  model_name TEXT,
                  city TEXT,
                  temperature REAL,
                  humidity REAL,
                  wind_speed REAL,
                  prediction REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS activity_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  activity TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

def update_db_schema():
    """Ensure database schema is up-to-date with home_city column"""
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN home_city TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    finally:
        conn.close()


# Database Operations
def create_user(username, password, email, home_city):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, email, home_city) VALUES (?, ?, ?, ?)",
                  (username, hashed_password, email, home_city))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def check_credentials(username, password):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, hashed_password))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None


def save_prediction(user_id, model_name, city, temperature, humidity, wind_speed, prediction):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''INSERT INTO predictions
                 (user_id, model_name, city, temperature, humidity, wind_speed, prediction)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
              (user_id, model_name, city, temperature, humidity, wind_speed, prediction))
    conn.commit()
    conn.close()


def log_activity(user_id, activity):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO activity_logs (user_id, activity) VALUES (?, ?)", (user_id, activity))
    conn.commit()
    conn.close()


def get_user_predictions(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM predictions WHERE user_id=?", (user_id,))
    results = c.fetchall()
    conn.close()
    return results


def get_all_users():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT id, username, email, home_city, created_at FROM users")
    results = c.fetchall()
    conn.close()
    return results


def delete_user(user_id):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM activity_logs WHERE user_id=?", (user_id,))
        c.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error deleting user: {str(e)}")
        return False
    finally:
        conn.close()


# Model Loading
@st.cache_resource
def load_rf_model():
    return joblib.load("forest_fire_rf_model.pkl")


@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("forest_fire_nn_model.h5", custom_objects={"mse": MeanSquaredError()})


# Email Alert Function


def send_email_alert(user_email, home_city, prediction):
    """
    Sends an email alert using Mailgun's API.
    """
    # Email Content
    subject = "🔥 High Fire Risk Alert for Your Home City"
    body = f"""
    <h2>High Fire Risk Detected in {home_city}!</h2>
    <p>Predicted Fire Risk: {prediction:.2f} hectares</p>
    <p>Please take necessary precautions and visit the app for more details.</p>
    <p><a href='http://your-streamlit-app-url.com'>View Details</a></p>
    """

    # Send Email via Mailgun API
    response = requests.post(
        f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
        auth=("api", MAILGUN_API_KEY),
        data={
            "from": f"Fire Prediction System <postmaster@{MAILGUN_DOMAIN}>",
            "to": user_email,
            "subject": subject,
            "html": body,  # Use "html" for rich formatting
        },
    )

    # Check Response
    if response.status_code == 200:
        st.success("✅ Email alert sent successfully!")
    else:
        st.error(f"❌ Failed to send email. Status Code: {response.status_code}")
        st.error(f"Response: {response.text}")


# Authentication Form
def login_register_form():
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <h1 style="margin: 0; padding-right: 10px;">🔥</h1>
                <h2 style="margin: 0;">Fire Prediction</h2>
            </div>
        """, unsafe_allow_html=True)
       
        st.subheader("Authentication")
        choice = st.radio("Choose Action", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")


        if choice == "Register":
            email = st.text_input("Email")
            home_city = st.text_input("Home City")
            if st.button("Register"):
                if create_user(username, password, email, home_city):
                    st.success("Registration successful! Please login.")
                    log_activity(1, "New user registration")
                else:
                    st.error("Username already exists!")
        else:
            if st.button("Login"):
                user_id = check_credentials(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    log_activity(user_id, "User login")
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials!")


        if "user_id" in st.session_state:
            if st.button("Logout"):
                log_activity(st.session_state.user_id, "User logout")
                del st.session_state.user_id
                del st.session_state.username
                st.rerun()


# Prediction Page
def predict_fire_page():
    st.subheader("Predict Fire Risk")
    model_choice = st.radio("🧠 Select Model", ("Random Forest", "Neural Network"))
    model = load_rf_model() if model_choice == "Random Forest" else load_nn_model()


    col1, col2 = st.columns(2)
    with col1:
        X = st.number_input("📍 X Coordinate", 0, 9)
        Y = st.number_input("📍 Y Coordinate", 0, 9)
        FFMC = st.number_input("🔥 FFMC Index", 0.0, 100.0)
        DMC = st.number_input("🔥 DMC Index", 0.0, 300.0)
        DC = st.number_input("🔥 DC Index", 0.0, 800.0)
        ISI = st.number_input("💨 ISI Index", 0.0, 50.0)
    with col2:
        temperature = st.number_input("🌡 Temperature (°C)", -10.0, 50.0)
        humidity = st.number_input("💧 Humidity (%)", 0, 100)
        wind_speed = st.number_input("🌬 Wind Speed (km/h)", 0.0, 100.0)
        rain = st.number_input("🌧 Rainfall (mm)", 0.0, 50.0)
        month = st.selectbox("🗓 Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        day = st.selectbox("🗓 Day", ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'])


    if st.button("🚀 Predict Fire Risk"):
        month_encoded = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].index(month)
        day_encoded = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'].index(day)
        input_data = np.array([[X, Y, FFMC, DMC, DC, ISI, temperature, humidity, wind_speed, rain, month_encoded, day_encoded]])


        try:
            prediction = model.predict(input_data)
            pred_value = prediction[0][0] if model_choice == "Neural Network" else prediction[0]
            st.session_state.user_inputs = {
                "X": X, "Y": Y, "FFMC": FFMC, "DMC": DMC, "DC": DC, "ISI": ISI,
                "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
                "rain": rain, "month_encoded": month_encoded, "day_encoded": day_encoded
            }
            st.success(f"🔥 Predicted Fire Spread: {pred_value:.2f} hectares")
            save_prediction(st.session_state.user_id, model_choice, "Manual Input", temperature, humidity, wind_speed, pred_value)
            log_activity(st.session_state.user_id, f"Prediction using {model_choice}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# Visualization Page
def visualizations_page():
    st.title("📊 Data Visualizations")
    if "user_inputs" not in st.session_state:
        st.warning("Please make a prediction first!")
        return


    df_user = pd.DataFrame([st.session_state.user_inputs])
    st.subheader("Input Features Analysis")
    with st.expander("View Feature Distribution"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=df_user.columns, y=df_user.iloc[0], palette="viridis")
        plt.xticks(rotation=45)
        st.pyplot(fig)


    st.subheader("Environmental Relationships")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=df_user["temperature"], y=df_user["FFMC"], color="red")
        ax1.set_title("Temperature vs FFMC")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df_user["humidity"], y=df_user["DMC"], color="blue")
        ax2.set_title("Humidity vs DMC")
        st.pyplot(fig2)


    st.subheader("Additional Visualizations")
    show_heatmap = st.checkbox("Show Correlation Heatmap")
    show_pairplot = st.checkbox("Show Pair Plot (May be slow)")


    if show_heatmap or show_pairplot:
        df = pd.read_csv("data/forestfires.csv")
        df_numeric = df.select_dtypes(include=[float, int])
        if show_heatmap:
            st.subheader("Feature Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm")
            st.pyplot(fig3)
        if show_pairplot:
            st.subheader("Pairwise Relationships")
            st.warning("This visualization may take a moment to load...")
            pair_grid = sns.pairplot(df_numeric)
            st.pyplot(pair_grid.fig)


# History Page
def my_history_page():
    st.subheader("Prediction History")
    predictions = get_user_predictions(st.session_state.user_id)
    if predictions:
        for pred in predictions:
            pred = list(pred)
            for i in range(len(pred)):
                if isinstance(pred[i], bytes):
                    try:
                        pred[i] = pred[i].decode('utf-8')
                    except:
                        pred[i] = str(pred[i])
            try:
                prediction_value = float(pred[7])
                prediction_str = f"{prediction_value:.2f}"
            except (ValueError, TypeError):
                prediction_str = str(pred[7])
            try:
                timestamp = datetime.strptime(pred[8], "%Y-%m-%d %H:%M:%S").strftime("%d %b %Y, %H:%M")
            except:
                timestamp = str(pred[8])
            with st.expander(f"Prediction #{pred[0]} - {timestamp}"):
                st.markdown(f"""
                    **Model:** {pred[2]}
                    **Location:** {pred[3]}
                    **Temperature:** {pred[4]}°C
                    **Humidity:** {pred[5]}%
                    **Wind Speed:** {pred[6]} km/h
                    **Predicted Area:** {prediction_str} hectares
                    **Date:** {timestamp}
                """)
    else:
        st.info("No predictions made yet!")


# Updated Admin Panel with Home City Monitoring
def admin_panel():
    st.subheader("Admin Dashboard")
    tab1, tab2, tab3 = st.tabs(["User Management", "System Monitoring", "Monitor Home Cities"])

    # User Management Tab (Unchanged)
    with tab1:
        st.write("### Registered Users")
        users = get_all_users()
        user_df = pd.DataFrame(users, columns=["ID", "Username", "Email", "Home City", "Join Date"])
        st.dataframe(user_df)

        st.write("### Delete User")
        user_to_delete = st.selectbox("Select User to Delete", [user[1] for user in users])
        if st.button("Delete User"):
            user_id_to_delete = next(user[0] for user in users if user[1] == user_to_delete)
            if delete_user(user_id_to_delete):
                st.success(f"User '{user_to_delete}' deleted successfully!")
                st.rerun()

    # System Monitoring Tab (Unchanged)
    with tab2:
        st.write("### Activity Logs")
        conn = sqlite3.connect(DATABASE_NAME)
        logs = conn.execute("SELECT * FROM activity_logs").fetchall()
        log_df = pd.DataFrame(logs, columns=["ID", "User ID", "Activity", "Timestamp"])
        st.dataframe(log_df)

    # Enhanced Home City Monitoring Tab
    with tab3:
        st.write("### Monitor Home Cities for Fire Risks")
        users = get_all_users()

    with tab3:
        st.write("### Monitor Home Cities for Fire Risks")
        users = get_all_users()

        if not users:
            st.info("No users with home cities found.")
            return

        model = load_nn_model()  # Default to neural network model

        # Add a slider for risk threshold and store its value in session state
        if "risk_threshold" not in st.session_state:
            st.session_state.risk_threshold = 0.5  # Default value

        # Update the slider value in session state when changed
        st.session_state.risk_threshold = st.slider(
            "Risk Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.risk_threshold, 
            step=0.01,
            key="risk_threshold_slider"
        )

        for user in users:
            user_id, username, email, home_city, _ = user
            if not home_city:
                continue

            with st.expander(f"{username}'s Home City: {home_city}"):
                lat, lon = get_city_coordinates(home_city)
                if not lat or not lon:
                    st.error(f"Could not find coordinates for {home_city}")
                    continue

                weather_data = fetch_weather_data(lat, lon)
                if not weather_data:
                    st.error(f"Failed to fetch weather data for {home_city}")
                    continue

                # Debug: Print weather data
                st.write("Weather Data:", weather_data)

                try:
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

                    # Prepare input data
                    input_data = np.array([[
                        lat, lon, ffmc, dmc, dc, isi,
                        weather_data["temperature"],
                        weather_data["humidity"],
                        weather_data["wind_speed"],
                        weather_data["rainfall"],
                        0, 0  # Placeholder for month/day
                    ]])

                    # Make prediction
                    raw_prediction = model.predict(input_data)[0][0]
                    probability = np.clip(raw_prediction, 0.0, 1.0)  # Clip to [0, 1]
                    risk_level = "🔥 HIGH RISK" if probability > st.session_state.risk_threshold else "✅ LOW RISK"

                    # Display results
                    cols = st.columns([3, 1])
                    cols[0].metric("Predicted Fire Risk", f"{probability:.2%}", risk_level)

                    # Alert system
                    if probability > st.session_state.risk_threshold:
                        cols[1].warning("Alert Condition Met!")
                        if cols[1].button(f"Notify {username}", key=f"alert_{user_id}"):
                            send_email_alert(email, home_city, probability)
                            st.success(f"Alert sent to {email}")

                except Exception as e:
                    st.error(f"Error processing {home_city}: {str(e)}")

 #Updated Main Function
def main():
    # Initialize the database
    init_db()
    update_db_schema()  # Ensure database schema is current

    # Show login/register form if the user is not logged in
    if "user_id" not in st.session_state or st.session_state.user_id is None:
        login_register_form()

        # Welcome page styling (only shown when not logged in)
        st.markdown("""
            <style>
            .stApp {
                background-image: url('https://images.unsplash.com/photo-1615092296061-e2ccfeb2f3d6');
                background-size: cover;
            }
            .welcome-header {
                color: white;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
                padding: 2rem;
                border-radius: 10px;
                background-color: rgba(0, 0, 0, 0.5);
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="welcome-header">
                <h1>🌲 Forest Fire Prediction System</h1>
                <p>This application helps predict and analyze forest fire risks using environmental parameters and machine learning models.</p>
            </div>
        """, unsafe_allow_html=True)
        return  # Stop further execution if the user is not logged in

    # Logged-in user interface
    menu_options = ["Make Prediction", "Data Visualizations", "My History"]
    if st.session_state.username == "admin":
        menu_options.append("Admin Panel")

    # Sidebar menu for navigation
    choice = st.sidebar.selectbox("Menu", menu_options)
    
    # Render the selected page
    if choice == "Make Prediction":
        st.write("If you want a prediction report and comparison of both models, go to the **Predict Fire** page from the sidebar.")
        predict_fire_page()
    elif choice == "Data Visualizations":
        visualizations_page()
    elif choice == "My History":
        my_history_page()
    elif choice == "Admin Panel":
        admin_panel()

    # Logout button in the sidebar
    if st.sidebar.button("Logout"):
        log_activity(st.session_state.user_id, "User logout")
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()  # Rerun the app to reset the state

if __name__ == "__main__":
    main()
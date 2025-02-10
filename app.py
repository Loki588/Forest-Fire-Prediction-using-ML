import streamlit as st
import sqlite3
import hashlib
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from tensorflow.keras.losses import MeanSquaredError
from fpdf import FPDF

# Database Configuration
DATABASE_NAME = "fire_prediction.db"

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT,
                  email TEXT,
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

# Database Operations
def create_user(username, password, email):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  (username, hashed_password, email))
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
    c.execute("SELECT id, username, email, created_at FROM users")
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

# Authentication Form
def login_register_form():
    with st.sidebar:
        # Sidebar header with icon
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
        email = st.text_input("Email") if choice == "Register" else ""

        if st.button(choice):
            if choice == "Register":
                if create_user(username, password, email):
                    st.success("Registration successful! Please login.")
                    log_activity(1, "New user registration")
                else:
                    st.error("Username already exists!")
            else:
                user_id = check_credentials(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    log_activity(user_id, "User login")
                    st.success("Logged in successfully!")
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

# Admin Panel
def admin_panel():
    st.subheader("Admin Dashboard")
    tab1, tab2 = st.tabs(["User Management", "System Monitoring"])

    with tab1:
        st.write("### Registered Users")
        users = get_all_users()
        user_df = pd.DataFrame(users, columns=["ID", "Username", "Email", "Join Date"])
        st.dataframe(user_df)

        st.write("### Delete User")
        user_to_delete = st.selectbox("Select User to Delete", [user[1] for user in users])
        if st.button("Delete User"):
            user_id_to_delete = next(user[0] for user in users if user[1] == user_to_delete)
            if delete_user(user_id_to_delete):
                st.success(f"User '{user_to_delete}' deleted successfully!")
                st.rerun()
            else:
                st.error("Failed to delete user.")

    with tab2:
        st.write("### Activity Logs")
        conn = sqlite3.connect(DATABASE_NAME)
        logs = conn.execute("SELECT * FROM activity_logs").fetchall()
        log_df = pd.DataFrame(logs, columns=["ID", "User ID", "Activity", "Timestamp"])
        st.dataframe(log_df)

# Main Application
def main():
    st.set_page_config(page_title="Forest Fire Prediction", layout="wide")
    init_db()

    with st.sidebar:
         st.markdown("""
            <style>
                .stLogo {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .stLogo img {
                    width: 600px !important;  /* Adjust the width as needed */
                    height: auto !important;  /* Maintain aspect ratio */
                    max-width: 100% !important;  /* Ensure it doesn't overflow */
                }
            </style>
        """, unsafe_allow_html=True)


        # Add logo to sidebar
         st.logo(
            image="forest logo.png",  # Replace with your logo URL
            size="large",
            link=None, # Optional website link
            icon_image=None
        )

    


    # Add background image and welcome message
    if "user_id" not in st.session_state:
        st.markdown(
            """
            <style>
            .stApp {
                background-image: url('https://images.unsplash.com/photo-1615092296061-e2ccfeb2f3d6');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
            .welcome-header {
                color: white;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
                padding: 2rem;
                border-radius: 10px;
                background-color: rgba(0, 0, 0, 0.5);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="welcome-header">
                <h1>🌲 Forest Fire Prediction System</h1>
                <p>This application helps predict and analyze forest fire risks using environmental parameters and machine learning models. 
                Please login or register to access prediction features.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    login_register_form()

    if "user_id" not in st.session_state:
        return

    menu_options = ["Make Prediction", "Data Visualizations", "My History"]
    if st.session_state.username == "admin":
        menu_options.append("Admin Panel")

    choice = st.sidebar.selectbox("Menu", menu_options)
    if choice == "Make Prediction":
        predict_fire_page()
    elif choice == "Data Visualizations":
        visualizations_page()
    elif choice == "My History":
        my_history_page()
    elif choice == "Admin Panel":
        admin_panel()

if __name__ == "__main__":
    main()
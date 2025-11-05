import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import st_folium
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://johnshirley2922_db_user:Shirley12345@cluster0b.vnr7bho.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0b"
DB_NAME = "UrbanSafe"
COLLECTION_NAME = "accidents"

# Set page configuration
st.set_page_config(
    page_title="UrbanSafe-AI Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #ff6b6b;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
@st.cache_data
def load_data():
    """Load and prepare collision data from MongoDB (fallback to CSV)"""
    try:
        # 🔗 Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Fetch all documents
        data = list(collection.find())
        if len(data) == 0:
            st.warning("⚠️ No data found in MongoDB collection. Falling back to CSV...")
            raise ValueError("Empty collection")

        df = pd.DataFrame(data)

        # Drop the internal MongoDB ID column
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        # Parse time and date columns
        df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
        df["day_of_week"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce").dt.dayofweek
        df = df.fillna(df.mode().iloc[0])

        st.success("✅ Data successfully loaded from MongoDB!")
        return df

    except Exception as e:
        st.error(f"❌ MongoDB connection failed: {e}")
        st.warning("⚠️ Loading from CSV instead...")

        # Fallback CSV load
        try:
            df = pd.read_csv("data/raw/dft-road-casualty-statistics-collision-2023.csv", low_memory=False)
            df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
            df["day_of_week"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce").dt.dayofweek
            df = df.fillna(df.mode().iloc[0])
            st.success("✅ Data loaded from local CSV as fallback.")
            return df
        except FileNotFoundError:
            st.error("🚫 No CSV or MongoDB data available!")
            return None


# Cache model training
@st.cache_resource
def train_model(data):
    """Train the accident prediction model"""
    features = ['hour', 'weather_conditions', 'road_type', 'speed_limit', 
               'light_conditions', 'road_surface_conditions', 'day_of_week']
    
    X = data[features]
    y = data['accident_severity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    return model, accuracy, X_test, y_test

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 UrbanSafe-AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Smart City Mobility & Accident Risk Prediction System</p>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Train model
    model, accuracy, X_test, y_test = train_model(data)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox("Choose Analysis", 
                               ["📊 Overview", "🔮 Risk Predictor", "🗺️ Hotspot Map", "📈 Analytics"])
    
    if page == "📊 Overview":
        show_overview(data, accuracy)
    elif page == "🔮 Risk Predictor":
        show_predictor(model)
    elif page == "🗺️ Hotspot Map":
        show_map(data)
    elif page == "📈 Analytics":
        show_analytics(data)

def show_overview(data, accuracy):
    """Display overview statistics"""
    st.header("📊 Project Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accidents", f"{len(data):,}")
    with col2:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    with col3:
        fatal_count = len(data[data['accident_severity'] == 1])
        st.metric("Fatal Accidents", f"{fatal_count:,}")
    with col4:
        peak_hour = data['hour'].mode()[0]
        st.metric("Peak Hour", f"{peak_hour}:00")
    
    # Severity Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accident Severity Distribution")
        severity_counts = data['accident_severity'].value_counts()
        severity_labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        
        fig = px.pie(
            values=severity_counts.values,
            names=[severity_labels.get(i, f'Level {i}') for i in severity_counts.index],
            color_discrete_sequence=['#ff6b6b', '#ffa726', '#66bb6a']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Accidents by Hour")
        hourly_data = data.groupby('hour').size().reset_index(name='count')
        
        fig = px.bar(
            hourly_data, 
            x='hour', 
            y='count',
            color='count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, xaxis_title="Hour of Day", yaxis_title="Number of Accidents")
        st.plotly_chart(fig, use_container_width=True)

def show_predictor(model):
    """Interactive accident risk predictor"""
    st.header("🔮 Accident Risk Predictor")
    st.write("Input scenario details to predict accident severity risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour of Day", 0, 23, 17, help="24-hour format")
        weather = st.selectbox("Weather Conditions", 
            ["Fine, no high winds", "Raining, no high winds", "Other", "Unknown"])
        road_type = st.selectbox("Road Type", 
            ["Roundabout", "One way street", "Dual carriageway", "Single carriageway", "Slip road"])
        
    with col2:
        speed_limit = st.selectbox("Speed Limit", [20, 30, 40, 50, 60, 70])
        light_conditions = st.selectbox("Light Conditions", 
            ["Daylight", "Darkness - lights lit", "Darkness - lights unlit"])
        surface = st.selectbox("Road Surface", ["Dry", "Wet or damp"])
        day_of_week = st.selectbox("Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    # Predict button
    if st.button("🚨 Predict Risk Level", type="primary"):
        # Map selections to codes
        weather_map = {"Fine, no high winds": 1, "Raining, no high winds": 2, "Other": 8, "Unknown": 9}
        road_map = {"Roundabout": 1, "One way street": 2, "Dual carriageway": 3, "Single carriageway": 6, "Slip road": 7}
        light_map = {"Daylight": 1, "Darkness - lights lit": 4, "Darkness - lights unlit": 5}
        surface_map = {"Dry": 1, "Wet or damp": 2}
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        weather_code = weather_map[weather]
        road_code = road_map[road_type]
        light_code = light_map[light_conditions]
        surface_code = surface_map[surface]
        day_code = day_map[day_of_week]
        
        # Make prediction
        features = np.array([[hour, weather_code, road_code, speed_limit, light_code, surface_code, day_code]])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Display results
        severity_names = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
        predicted_severity = severity_names[prediction]
        
        # Color coding
        colors = {1: '#ff4444', 2: '#ff8800', 3: '#44ff44'}
        color = colors[prediction]
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 2rem; border-radius: 1rem; text-align: center; margin: 1rem 0;">
            <h2 style="color: {color}; margin: 0;">Predicted Severity: {predicted_severity}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.subheader("Risk Breakdown")
        prob_df = pd.DataFrame({
            'Severity': ['Fatal', 'Serious', 'Slight'],
            'Probability': probabilities
        })
        
        fig = px.bar(prob_df, x='Severity', y='Probability', 
                     color='Probability', color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_map(data):
    """Display accident hotspot map"""
    st.header("🗺️ Accident Hotspot Map")
    
    # Filter data with valid coordinates
    map_data = data.dropna(subset=['latitude', 'longitude'])
    map_data = map_data[(map_data['latitude'] != 0) & (map_data['longitude'] != 0)]
    
    if len(map_data) == 0:
        st.error("No valid coordinate data available for mapping")
        return
    
    # Sample data for performance
    if len(map_data) > 1000:
        map_data = map_data.sample(1000)
    
    # Severity filter
    severity_filter = st.selectbox("Filter by Severity", 
                                  ["All", "Fatal", "Serious", "Slight"])
    
    if severity_filter != "All":
        severity_map = {"Fatal": 1, "Serious": 2, "Slight": 3}
        map_data = map_data[map_data['accident_severity'] == severity_map[severity_filter]]
    
    # Create map
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Add markers
    colors = {1: 'red', 2: 'orange', 3: 'green'}
    for idx, row in map_data.iterrows():
        color = colors.get(row['accident_severity'], 'blue')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            color=color,
            fillColor=color,
            weight=1,
            opacity=0.8
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)
    
    st.info(f"Showing {len(map_data)} accidents on map")

def show_analytics(data):
    """Advanced analytics dashboard"""
    st.header("📈 Advanced Analytics")
    
    # Time series analysis
    st.subheader("Monthly Accident Trends")
    data['month'] = pd.to_datetime(data['date'], format='%d/%m/%Y').dt.month
    monthly_data = data.groupby('month').size().reset_index(name='count')
    
    fig = px.line(monthly_data, x='month', y='count', markers=True)
    fig.update_layout(xaxis_title="Month", yaxis_title="Number of Accidents")
    st.plotly_chart(fig, use_container_width=True)
    
    # Weather impact analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Weather vs Severity")
        weather_severity = pd.crosstab(data['weather_conditions'], data['accident_severity'])
        weather_labels = {1: 'Fine', 2: 'Raining', 8: 'Other', 9: 'Unknown'}
        
        fig = px.imshow(weather_severity.values, 
                       x=['Fatal', 'Serious', 'Slight'],
                       y=[weather_labels.get(i, f'Condition {i}') for i in weather_severity.index],
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Day of Week Patterns")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_accidents = data.groupby('day_of_week').size().reset_index(name='count')
        daily_accidents['day_name'] = daily_accidents['day_of_week'].map(lambda x: day_names[x])
        
        fig = px.bar(daily_accidents, x='day_name', y='count', color='count')
        fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Accidents")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
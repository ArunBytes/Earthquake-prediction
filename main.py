import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Earthquake Predictor", page_icon="🌍", layout="wide")

# --- PREMIUM UI STYLING (Glassmorphism & Gradients) ---
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Transparent Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Clean Top Navbar */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Styled Submit Button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        border: none;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        color: white;
        border: none;
    }
    
    /* Prediction Card (Glassmorphism) */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        margin-top: 1rem;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: slideUp 0.5s ease-out forwards;
    }
    
    /* Prediction Value Gradient text */
    .pred-value {
        font-size: 3rem;
        font-weight: 900;
        background: -webkit-linear-gradient(#f12711, #f5af19);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .pred-title {
        font-size: 1.5rem;
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    .pred-subtitle {
        color: #a0a0a0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Keyframes for subtle load animation */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL TRAINING AND LOADING ---
@st.cache_resource(show_spinner="Initializing Machine Learning Models (Random Forest)...")
def load_and_train_model():
    """
    Loads dataset, preprocesses, and trains the Random Forest exactly mimicking the EDA notebook.
    Uses st.cache_resource so it only executes when the app spins up.
    """
    try:
        # First attempt local workspace path
        df = pd.read_csv('Dataset/Earthquake_Data.csv', delimiter=r'\s+')
    except FileNotFoundError:
        # Fallback to C:/content/ if that's where the dataset lives
        df = pd.read_csv('C:/content/Earthquake_Data.csv', delimiter=r'\s+')
    
    new_column_names = ["Date", "Time", "Latitude", "Longitude", "Depth", "Magnitude", 
                        "Magnitude_type", "No_of_Stations", "Gap", "Close", "RMS", "SRC", "EventID"]
    df.columns = new_column_names
    
    # Feature extraction exactly as done in EDA_J_Component.py
    X = df[['Latitude', 'Longitude', 'Depth', 'No_of_Stations']]
    y = df['Magnitude']
    
    # Train the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Train SVR (using subset of 500 rows to ensure fast loading times)
    subset_size = min(500, len(X))
    X_subset = X[:subset_size]
    y_subset = y[:subset_size]
    svm_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svm_model.fit(X_subset, y_subset)
    
    # Pre-calculate predictions over the dataset for charting
    rf_pred = rf.predict(X)
    
    return rf, lr, svm_model, X, y, rf_pred

# Load the model
try:
    rf_model, lr_model, svm_model, X_ref, y_ref, rf_pred_ref = load_and_train_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load the model or dataset: {e}")

# --- APP LAYOUT ---
if model_loaded:
    st.title("🌋 Seismic Magnitude Analyzer")
    st.markdown("Harnessing **Machine Learning (Random Forest)** to predict the Richter magnitude of an Earthquake event based on geological factors and seismic array detection.")

    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("Geospatial Input Data")
        st.markdown("Adjust the parameters to simulate a seismic event.")
        
        # Ranges based roughly on California Earthquake data (dataset scope)
        lat_input = st.slider("Latitude (deg)", float(X_ref['Latitude'].min()), float(X_ref['Latitude'].max()), float(X_ref['Latitude'].mean()))
        lon_input = st.slider("Longitude (deg)", float(X_ref['Longitude'].min()), float(X_ref['Longitude'].max()), float(X_ref['Longitude'].mean()))
        
        depth_input = st.number_input("Depth (km)", min_value=0.0, max_value=1000.0, value=float(X_ref['Depth'].mean()), step=1.0)
        stations_input = st.number_input("No. of Recording Stations", min_value=0, max_value=500, value=int(X_ref['No_of_Stations'].mean()), step=1)
        
        st.markdown("---")
        predict_btn = st.button("Calculate Magnitude ⚡")
        
        st.markdown("<br><br><small>Built with 🤍 using Streamlit</small>", unsafe_allow_html=True)

    # MAIN AREA TABS
    tab1, tab2 = st.tabs(["🌍 Dynamic Predictor", "📊 Graphs & Image Analytics"])
    
    with tab1:
        # Interactive Map Visualization
        st.markdown("### Epicenter Location")
        map_df = pd.DataFrame({'lat': [lat_input], 'lon': [lon_input]})
        st.map(map_df, zoom=5, color="#ff4b2b")
    
        # Prediction Logic
        if predict_btn:
            with st.spinner("Analyzing seismic telemetry..."):
                # Prepare feature vector [Latitude, Longitude, Depth, No_of_Stations]
                input_features = pd.DataFrame([[lat_input, lon_input, depth_input, stations_input]], 
                                              columns=['Latitude', 'Longitude', 'Depth', 'No_of_Stations'])
                
                pred_rf = rf_model.predict(input_features)[0]
                pred_lr = lr_model.predict(input_features)[0]
                pred_svm = svm_model.predict(input_features)[0]
                
                # Construct rich UI cards side-by-side
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="pred-title">Random Forest</div>
                        <div class="pred-value">{pred_rf:.2f}</div>
                        <div class="pred-subtitle">Highest Accuracy Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="pred-title">Linear Regression</div>
                        <div class="pred-value">{pred_lr:.2f}</div>
                        <div class="pred-subtitle">Baseline Model</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div class="pred-title">SVM (SVR)</div>
                        <div class="pred-value">{pred_svm:.2f}</div>
                        <div class="pred-subtitle">Support Vector Regressor</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("👈 Please define the geospatial event metrics in the sidebar and click **Calculate Magnitude**.")

    with tab2:
        st.header("EDA & Model Evaluation Metrics")
        st.markdown("Mathematical charts generated directly from our Random Forest inference loop.")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Feature Importance")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            importances = rf_model.feature_importances_
            features = ['Latitude', 'Longitude', 'Depth', 'No. of Stations']
            if len(importances) == 3: features = features[:3]
            ax1.bar(features, importances, color=['#ff4b2b', '#ff416c', '#f5af19', '#f12711'])
            ax1.set_ylabel('Importance Weighting')
            st.pyplot(fig1)
            
            st.subheader("Actual vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y_ref, rf_pred_ref, alpha=0.3, color='#4CAF50')
            ax2.plot([y_ref.min(), y_ref.max()], [y_ref.min(), y_ref.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual Dataset Magnitude')
            ax2.set_ylabel('Model Predicted Magnitude')
            st.pyplot(fig2)
            
        with c2:
            st.subheader("Residual Plot")
            fig3 = plt.figure(figsize=(6, 4))
            sns.residplot(x=y_ref, y=rf_pred_ref, color='orange', scatter_kws={'alpha': 0.3})
            plt.xlabel('Predicted Magnitude Range')
            plt.ylabel('Residual Error Margin')
            st.pyplot(fig3)
            
            st.subheader("Support Vector Boundaries")
            try:
                st.image("images/SVM_plot.png", caption="Local SVM Model Data Plot", use_container_width=True)
            except Exception:
                st.warning("SVM_plot.png not found locally.")

        st.markdown("---")
        st.header("Tableau Visualizations")
        st.markdown("Externally sourced data-distributions from the project's foundational research.")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image("https://user-images.githubusercontent.com/113085803/229277065-c75d240b-1b1c-4a42-a43a-53311275cdf0.png", caption="Figure 1: Earthquakes by Stations recording it")
            st.image("https://user-images.githubusercontent.com/113085803/229277078-5e2a9ab0-10af-43f6-a862-7727770659ec.png", caption="Figure 3: Earthquake magnitude types")
        with col_img2:
            st.image("https://user-images.githubusercontent.com/113085803/229278389-8b9951d6-a06b-4316-bed9-fa220b5170a8.png", caption="Figure 2: Earthquake occurrences by magnitude rating")
            st.image("https://user-images.githubusercontent.com/113085803/229277278-88821402-2d5d-4beb-8644-e2b0b9bd73c0.png", caption="Figure 4: Magnitude and depth trends over time")

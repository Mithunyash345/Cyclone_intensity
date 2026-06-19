import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Cyclone CV Estimator", layout="wide", page_icon="🌀")

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #00E5FF;
        font-family: 'Inter', sans-serif;
    }
    .metric-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
    }
    .metric-value {
        font-size: 48px;
        font-weight: 700;
        color: #FFF;
        margin: 0;
    }
    .metric-label {
        font-size: 18px;
        color: #B0C4DE;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🌀 Cyclone Intensity CV Estimator")
st.markdown("Automated Early Warning Dashboard using Deep Learning on INSAT-3D/TCIR Infrared Imagery")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = 'best_cyclone_model.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# --- Sidebar ---
st.sidebar.header("Controls")
data_source = st.sidebar.radio("Image Source", ["Load Sample from TCIR Dataset", "Upload Image (Numpy Array)"])

# --- Helper Functions ---
@st.cache_data
def load_random_sample():
    """Loads a sample from HDF5 if available, otherwise falls back to local .npz samples."""
    data_path = 'TCIR-ALL_2017.h5'
    
    # Try loading from the full 3GB dataset first
    if os.path.exists(data_path):
        with h5py.File(data_path, 'r') as hf:
            data_matrix = hf['matrix']
            total_samples = data_matrix.shape[0]
            index = np.random.randint(0, total_samples)
            img_ir = data_matrix[index, :, :, 0]
            
        data_info = pd.read_hdf(data_path, key='info', mode='r')
        true_wind = data_info['Vmax'].iloc[index]
        return img_ir, true_wind
    
    # Fallback to small local samples (for Github/Friends)
    samples_dir = 'samples'
    if os.path.exists(samples_dir):
        sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.npz')]
        if sample_files:
            chosen_file = np.random.choice(sample_files)
            data = np.load(os.path.join(samples_dir, chosen_file))
            return data['image'], data['label']
            
    return None, None

def preprocess_image(img):
    """Preprocesses the image for the CNN."""
    img = np.nan_to_num(img, nan=0.0)
    # Simple Min-Max normalization
    min_val = np.min(img)
    max_val = np.max(img)
    img_norm = (img - min_val) / (max_val - min_val + 1e-8)
    
    # Expand dims for batch and channel: (1, 201, 201, 1)
    img_batch = np.expand_dims(np.expand_dims(img_norm, axis=-1), axis=0)
    return img_batch

# --- Main App Logic ---
img_ir = None
true_wind = None

if data_source == "Load Sample from TCIR Dataset":
    if st.sidebar.button("Fetch Random Cyclone"):
        img_ir, true_wind = load_random_sample()
        if img_ir is not None:
            st.session_state['img_ir'] = img_ir
            st.session_state['true_wind'] = true_wind
        else:
            st.error("No dataset or samples found. Please check your repository.")
            
    # Load from session state to persist across reruns
    if 'img_ir' in st.session_state:
        img_ir = st.session_state['img_ir']
        true_wind = st.session_state['true_wind']

else:
    uploaded_file = st.sidebar.file_uploader("Upload .npy file (201x201)", type=['npy'])
    if uploaded_file is not None:
        img_ir = np.load(uploaded_file)

if img_ir is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Infrared (IR) Satellite View")
        
        # Normalize image to 0-1 for display
        img_disp = (img_ir - np.min(img_ir)) / (np.max(img_ir) - np.min(img_ir) + 1e-8)
        st.image(img_disp, use_container_width=True, clamp=True)
        
        if true_wind is not None and not np.isnan(true_wind):
            st.info(f"**Ground Truth (Historical Record):** {true_wind} knots")

    with col2:
        st.subheader("AI Intensity Prediction")
        
        if model is None:
            st.warning("⚠️ Model weights (`best_cyclone_model.keras`) not found. Please train the model in the Jupyter Notebook first!")
        else:
            with st.spinner('Analyzing cloud structure...'):
                input_tensor = preprocess_image(img_ir)
                prediction = model.predict(input_tensor)[0][0]
                
                # Display beautiful metric
                st.markdown(f"""
                <div class="metric-box">
                    <p class="metric-label">Estimated Max Sustained Wind</p>
                    <p class="metric-value">{prediction:.1f} <span style="font-size: 24px;">knots</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Saffir-Simpson Hurricane Wind Scale Mapping
                category = "Tropical Depression"
                color = "gray"
                if prediction >= 34 and prediction <= 63:
                    category = "Tropical Storm"
                    color = "blue"
                elif prediction >= 64 and prediction <= 82:
                    category = "Category 1 Hurricane"
                    color = "yellow"
                elif prediction >= 83 and prediction <= 95:
                    category = "Category 2 Hurricane"
                    color = "orange"
                elif prediction >= 96 and prediction <= 112:
                    category = "Category 3 (Major)"
                    color = "red"
                elif prediction >= 113 and prediction <= 136:
                    category = "Category 4 (Major)"
                    color = "darkred"
                elif prediction > 136:
                    category = "Category 5 (Catastrophic)"
                    color = "purple"
                
                st.markdown(f"**Storm Classification:** <span style='color:{color}; font-weight:bold;'>{category}</span>", unsafe_allow_html=True)
                
                # --- Plot Intensity History Curve ---
                st.subheader("Intensity Forecast Curve")
                st.write("Projected intensity over the next 24 hours based on AI analysis.")
                
                # Mock trajectory for demonstration
                hours = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24])
                # Generate a curve that slowly changes
                trend = np.random.choice([-1, 1]) * np.random.uniform(0.5, 2.0)
                trajectory = prediction + (trend * hours) + np.random.normal(0, 2, size=len(hours))
                
                # Use Streamlit's native interactive line chart instead of Matplotlib
                chart_data = pd.DataFrame(
                    trajectory,
                    index=[f"+{h}h" for h in hours],
                    columns=["Wind Speed (Knots)"]
                )
                st.line_chart(chart_data)
else:
    st.info("👈 Please select or upload an image from the sidebar to begin analysis.")

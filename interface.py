import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Cancer Prediction System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Custom CSS with soft, elegant design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Beautiful gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 24px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        letter-spacing: -1.5px;
        text-shadow: 0 4px 20px rgba(0,0,0,0.15);
        position: relative;
        z-index: 1;
    }
    
    .main-title * {
        color: #ffffff !important;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.95);
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        letter-spacing: 0.3px;
    }
    
    /* Beautiful card styling */
    .elegant-card {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .elegant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.15);
    }
    
    .feature-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.08);
        transition: all 0.3s ease;
    }
    
    .feature-section:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.12);
        border-left-color: #764ba2;
    }
    
    .feature-section h3 {
        color: #4a5568 !important;
        font-weight: 600;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        letter-spacing: -0.3px;
    }
    
    /* Section headers */
    h2 {
        color: #2d3748 !important;
        font-weight: 600;
        font-size: 1.9rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.5px;
    }
    
    /* Beautiful prediction box */
    .prediction-box {
        padding: 3rem;
        border-radius: 24px;
        background: #ffffff;
        box-shadow: 0 15px 50px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
        text-align: center;
        border: 2px solid;
        position: relative;
        overflow: hidden;
        animation: resultFadeIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    @keyframes resultFadeIn {
        from {
            opacity: 0;
            transform: translateY(20px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    .prediction-benign {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-color: #48bb78;
        box-shadow: 0 15px 50px rgba(72, 187, 120, 0.25);
    }
    
    .prediction-benign::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .prediction-malignant {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border-color: #ed8936;
        box-shadow: 0 15px 50px rgba(237, 137, 54, 0.25);
    }
    
    .prediction-malignant::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.05); }
    }
    
    /* Beautiful button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1.1rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        letter-spacing: 0.3px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255,255,255,0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton>button:hover::before {
        width: 400px;
        height: 400px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
    }
    
    /* Beautiful metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    /* Elegant input fields */
    .stNumberInput input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        transition: all 0.3s ease;
        background: #ffffff;
        color: #2d3748;
        font-weight: 500;
    }
    
    .stNumberInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        background: #ffffff;
    }
    
    /* Beautiful sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7fafc 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 14px;
        border: none;
        box-shadow: 0 5px 20px rgba(0,0,0,0.06);
        animation: slideIn 0.4s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Beautiful info badge */
    .info-badge {
        display: inline-block;
        padding: 0.6rem 1.4rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.25);
        letter-spacing: 0.3px;
    }
    
    /* Model result cards */
    .model-result {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.2rem;
        border-bottom: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        border-radius: 10px;
        margin: 0.3rem 0;
    }
    
    .model-result:hover {
        background: rgba(102, 126, 234, 0.04);
        transform: translateX(8px);
    }
    
    .model-result:last-child {
        border-bottom: none;
    }
    
    .prediction-indicator {
        padding: 0.5rem 1.1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        letter-spacing: 0.3px;
    }
    
    .benign-indicator {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #22543d;
    }
    
    .malignant-indicator {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #742a2a;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 12px;
        font-weight: 600;
        color: #2d3748;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox enhancement */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        background: #ffffff;
        color: #2d3748;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* Caption styling */
    .stCaption {
        color: #718096;
        font-weight: 400;
    }
    
    /* Divider enhancement */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
    }
    
    /* Text colors */
    p, span, div {
        color: #4a5568;
    }
    
    label {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    /* Success/error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border: none;
        color: #22543d;
    }
    
    .stError {
        background: linear-gradient(135deg, #fab1a0 0%, #ffeaa7 100%);
        border: none;
        color: #742a2a;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #ffeaa7 0%, #ffd89b 100%);
        border: none;
        color: #744210;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border: none;
        color: #2c5282;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Secondary button styling */
    .stButton>button[kind="secondary"] {
        background: linear-gradient(135deg, #cbd5e0 0%, #e2e8f0 100%);
        color: #2d3748;
        border: 1px solid #cbd5e0;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.5rem;
        background: #ffffff;
        border: 2px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }
    
    /* Smooth scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Feature names
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
    'smoothness_worst', 'compactness_worst', 'concavity_worst', 
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Default test values
DEFAULT_VALUES = [
    17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760, 0.3001, 0.14710, 0.2419, 0.07871,
    1.0950, 0.9053, 8.589, 153.40, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.60, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890
]

# Feature descriptions and ranges
FEATURE_INFO = {
    'mean': ('Mean Values', 'Average measurements of cell nuclei', '📊'),
    'se': ('Standard Error', 'Variability in measurements', '📈'),
    'worst': ('Worst Values', 'Largest/most severe measurements', '⚠️')
}

# Model information dictionary
MODEL_INFO = {
    'KNN': {
        'name': 'K-Nearest Neighbors',
        'description': 'Classifier using proximity-based learning with pattern recognition.',
        'icon': '📍'
    },
    'Logistic Regression': {
        'name': 'Logistic Regression',
        'description': 'Linear model for binary classification with probability estimates.',
        'icon': '📉'
    },
    'Neural Network (MLP)': {
        'name': 'Multi-Layer Perceptron',
        'description': 'Deep learning model with advanced pattern detection capabilities.',
        'icon': '🧠'
    },
    'Random Forest': {
        'name': 'Random Forest',
        'description': 'Ensemble of decision trees for robust and accurate predictions.',
        'icon': '🌳'
    },
    'SVM': {
        'name': 'Support Vector Machine',
        'description': 'Finds optimal hyperplane for classification with maximum margin.',
        'icon': '⚡'
    },
    'Gradient Boosting': {
        'name': 'Gradient Boosting',
        'description': 'Sequential ensemble model that builds trees to correct previous errors.',
        'icon': '🚀'
    },
    'XGBoost': {
        'name': 'XGBoost',
        'description': 'Optimized gradient boosting with regularization for high performance.',
        'icon': '💫'
    },
    'Ensemble Voting': {
        'name': 'Ensemble Voting Classifier',
        'description': 'Combines predictions from all models for more reliable results.',
        'icon': '🤝'
    }
}

# Initialize session state
if 'use_defaults' not in st.session_state:
    st.session_state.use_defaults = False

@st.cache_resource
def load_models():
    """Load all saved models and scaler using joblib"""
    models = {}
    try:
        # Load all individual models
        models['KNN'] = joblib.load('models/KNeighborsClassifier.pkl')
        models['Logistic Regression'] = joblib.load('models/LogisticRegression_softmax.pkl')
        models['Neural Network (MLP)'] = joblib.load('models/MLPClassifier.pkl')
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        models['SVM'] = joblib.load('models/SVM.pkl')
        models['Gradient Boosting'] = joblib.load('models/gradient_boosting.pkl')
        models['XGBoost'] = joblib.load('models/xgboost.pkl')
        
        # Load scaler
        models['scaler'] = joblib.load('models/scaler.pkl')
        
        st.success("✅ All models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all model files are in the 'models/' directory.")
        return None

def create_ensemble_prediction(models_dict, scaler, features):
    """Make predictions using all models and perform ensemble voting"""
    # Convert features to array in correct order
    feature_array = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    predictions = {}
    probabilities = {}
    
    # Get predictions from all individual models (excluding scaler and ensemble)
    model_keys = [key for key in models_dict.keys() if key != 'scaler']
    
    for model_name in model_keys:
        try:
            model = models_dict[model_name]
            pred = model.predict(scaled_features)[0]
            predictions[model_name] = pred
            
            # Get probabilities if available
            try:
                prob = model.predict_proba(scaled_features)[0]
                probabilities[model_name] = prob
            except:
                probabilities[model_name] = None
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")
            predictions[model_name] = None
    
    # Perform majority voting
    valid_predictions = [p for p in predictions.values() if p is not None]
    if valid_predictions:
        ensemble_prediction = np.round(np.mean(valid_predictions))
        
        # Calculate average probability across models that provide probabilities
        valid_probs = [prob for prob in probabilities.values() if prob is not None]
        if valid_probs:
            avg_probability = np.mean(valid_probs, axis=0)
        else:
            avg_probability = None
        
        return ensemble_prediction, avg_probability, predictions, probabilities
    else:
        return None, None, predictions, probabilities

def make_prediction(model, scaler, features):
    """Make prediction using a single model"""
    # Convert features to array in correct order
    feature_array = np.array([features[f] for f in FEATURE_NAMES]).reshape(1, -1)
    
    # Scale features
    scaled_features = scaler.transform(feature_array)
    
    # Make prediction
    try:
        prediction = model.predict(scaled_features)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(scaled_features)[0]
            return prediction, probability
        except:
            return prediction, None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def create_input_form():
    """Create organized input form for features"""
    features = {}
    
    # Group features by type
    feature_groups = {
        'mean': [f for f in FEATURE_NAMES if f.endswith('_mean')],
        'se': [f for f in FEATURE_NAMES if f.endswith('_se')],
        'worst': [f for f in FEATURE_NAMES if f.endswith('_worst')]
    }
    
    # Create mapping of feature names to default values
    default_dict = dict(zip(FEATURE_NAMES, DEFAULT_VALUES))
    
    for group_key, (group_name, group_desc, icon) in FEATURE_INFO.items():
        st.markdown(f"<div class='feature-section'>", unsafe_allow_html=True)
        st.markdown(f"### {icon} {group_name}")
        st.caption(group_desc)
        
        cols = st.columns(3)
        for idx, feature in enumerate(feature_groups[group_key]):
            with cols[idx % 3]:
                # Clean feature name for display
                display_name = feature.replace('_mean', '').replace('_se', '').replace('_worst', '').replace('_', ' ').title()
                
                # Use default value if use_defaults is True
                default_val = default_dict[feature] if st.session_state.use_defaults else 0.0
                
                features[feature] = st.number_input(
                    display_name,
                    min_value=0.0,
                    value=float(default_val),
                    step=0.01,
                    format="%.4f",
                    key=feature
                )
        st.markdown("</div>", unsafe_allow_html=True)
    
    return features

def main():
    # Header
    st.markdown("""
        <div class='main-header'>
            <h1 class='main-title'>🎗️ Breast Cancer Prediction System</h1>
            <p class='subtitle'>Advanced AI-Powered Medical Diagnosis Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models_dict = load_models()
    
    if models_dict is None:
        st.error("⚠️ Unable to load models. Please ensure all model files are in the same directory.")
        st.info("""
        Required model files:
        - KNeighborsClassifier.pkl
        - LogisticRegression_softmax.pkl
        - MLPClassifier.pkl
        - random_forest.pkl
        - SVM.pkl
        - gradient_boosting.pkl
        - xgboost.pkl
        - scaler.pkl
        """)
        return
    
    # Sidebar for model selection
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")
        
        # Model selection dropdown with all models including ensemble
        model_options = list(MODEL_INFO.keys())
        model_choice = st.selectbox(
            "🔬 Select Prediction Model",
            model_options,
            help="Choose the machine learning model for prediction"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        
        selected_info = MODEL_INFO[model_choice]
        st.markdown(f"""
        <div style="padding: 1.3rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%); border-radius: 14px; border: 1px solid rgba(102, 126, 234, 0.15);">
            <h4 style="margin: 0; color: #667eea; font-size: 1.1rem; font-weight: 600;">{selected_info['icon']} {selected_info['name']}</h4>
            <p style="margin: 0.8rem 0 0 0; color: #718096; font-size: 0.9rem; line-height: 1.6;">{selected_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show all available models
        with st.expander("🔍 All Available Models"):
            st.markdown("### Loaded Models:")
            for model_name in MODEL_INFO.keys():
                if model_name in models_dict or model_name == 'Ensemble Voting':
                    st.markdown(f"✅ **{model_name}**")
                else:
                    st.markdown(f"❌ {model_name}")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Load Sample", use_container_width=True):
                st.session_state.use_defaults = True
                st.rerun()
        with col2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.use_defaults = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="padding: 1.3rem; background: rgba(102, 126, 234, 0.04); border-radius: 14px; font-size: 0.9rem; border: 1px solid rgba(102, 126, 234, 0.1);">
            <p style="color: #4a5568; line-height: 1.7; margin: 0;">
            This system uses machine learning algorithms to analyze cell nuclei measurements and predict breast cancer diagnosis.
            </p>
            <br>
            <p style="color: #718096; margin: 0;"><strong style="color: #667eea;">⚕️ Medical Disclaimer</strong><br>
            This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2.5, 1.5])
    
    with col1:
        st.markdown("##  Patient Data Input")
        st.markdown("Enter the cell nuclei measurements obtained from diagnostic imaging:")
        
        # Create input form
        features = create_input_form()
    
    with col2:
        st.markdown("##  Analysis Results")
        
        # Predict button
        if st.button("🔍 Analyze & Predict", use_container_width=True):
            if all(v == 0.0 for v in features.values()):
                st.warning("⚠️ Please enter valid measurements before prediction.")
            else:
                with st.spinner("🔄 Analyzing data..."):
                    scaler = models_dict['scaler']
                    
                    if model_choice == 'Ensemble Voting':
                        # Use ensemble voting
                        ensemble_result, avg_probability, individual_preds, individual_probs = create_ensemble_prediction(
                            models_dict, scaler, features
                        )
                        
                        if ensemble_result is not None:
                            prediction = ensemble_result
                            probability = avg_probability
                            
                            # Display individual model predictions
                            st.markdown("### 🤝 Ensemble Voting Results")
                            
                            for model_name, pred in individual_preds.items():
                                if pred is not None:
                                    pred_text = "Benign" if pred == 0 else "Malignant"
                                    pred_class = "benign-indicator" if pred == 0 else "malignant-indicator"
                                    st.markdown(f"""
                                    <div class="model-result">
                                        <span style="color: #2d3748;"><strong>{model_name}</strong></span>
                                        <span class="prediction-indicator {pred_class}">{pred_text}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                    else:
                        # Use single model
                        selected_model = models_dict[model_choice]
                        prediction, probability = make_prediction(selected_model, scaler, features)
                    
                    if prediction is not None:
                        # Display results
                        box_class = "prediction-benign" if prediction == 0 else "prediction-malignant"
                        result_emoji = "✓" if prediction == 0 else "!"
                        result_text = "Benign" if prediction == 0 else "Malignant"
                        result_desc = "non-cancerous" if prediction == 0 else "cancerous"
                        
                        st.markdown(f"""
                        <div class='prediction-box {box_class}'>
                            <div style="font-size: 5rem; margin-bottom: 1rem; font-weight: 700; opacity: 0.9;">{result_emoji}</div>
                            <h2 style="margin: 0; font-size: 2.2rem; color: #2d3748; font-weight: 700;">{result_text} Diagnosis</h2>
                            <p style="margin: 1.2rem 0 0 0; font-size: 1.05rem; color: #4a5568;">
                                The model predicts a <strong style="color: #2d3748;">{result_desc}</strong> diagnosis
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if probability is not None:
                            st.markdown("###  Confidence Analysis")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Benign", f"{probability[0]*100:.1f}%")
                            with col_b:
                                st.metric("Malignant", f"{probability[1]*100:.1f}%")
                        
                        # Model info
                        if model_choice == 'Ensemble Voting':
                            model_display = f"Ensemble Voting ({len([p for p in individual_preds.values() if p is not None])} models)"
                        else:
                            model_display = model_choice
                        
                        st.markdown(f"<div class='info-badge'>Model: {model_display}</div>", unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.warning("⚕️ **Important**: This prediction should be reviewed by qualified medical professionals. Never rely solely on automated systems for medical decisions.")

if __name__ == "__main__":
    main()
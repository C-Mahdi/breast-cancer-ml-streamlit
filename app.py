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

# Custom CSS for elegant styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f8f9fa;
    }
    
    .stApp {
        background: #f8f9fa;
    }
    
    [data-testid="stAppViewContainer"] {
        background: #f8f9fa;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .main-title * {
        color: #ffffff !important;
    }
    
    .subtitle {
        color: #e0e0e0;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .elegant-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #e8e8e8;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .elegant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .feature-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #2C5364;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .feature-section h3 {
        color: #2C5364 !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Section headers */
    h2 {
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    /* Prediction box */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 16px;
        background: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        border: 2px solid #e8e8e8;
    }
    
    .prediction-benign {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-color: #10b981;
    }
    
    .prediction-malignant {
        background: linear-gradient(135deg, #fef3c715 0%, #fca5a515 100%);
        border-color: #ef4444;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2C5364 0%, #0F2027 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(44, 83, 100, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 83, 100, 0.4);
        background: linear-gradient(135deg, #0F2027 0%, #2C5364 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #2C5364;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Input fields */
    .stNumberInput input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
        transition: border-color 0.2s ease;
    }
    
    .stNumberInput input:focus {
        border-color: #2C5364;
        box-shadow: 0 0 0 2px rgba(44, 83, 100, 0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e8e8e8;
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Info badge */
    .info-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        background: #e8f4f8;
        color: #2C5364;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Ensemble voting display */
    .voting-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #2C5364;
    }
    
    .model-result {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e8e8e8;
    }
    
    .model-result:last-child {
        border-bottom: none;
    }
    
    .prediction-indicator {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .benign-indicator {
        background: #d1fae5;
        color: #065f46;
    }
    
    .malignant-indicator {
        background: #fee2e2;
        color: #991b1b;
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
    'mean': ('Mean Values', 'Average measurements of cell nuclei', ''),
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
        'icon': '⚡'
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
            " Select Prediction Model",
            model_options,
            help="Choose the machine learning model for prediction"
        )
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        
        selected_info = MODEL_INFO[model_choice]
        st.markdown(f"""
        **{selected_info['icon']} {selected_info['name']}**
        
        {selected_info['description']}
        """)
        
        # Show all available models
        with st.expander(" All Available Models"):
            st.markdown("### Loaded Models:")
            for model_name in MODEL_INFO.keys():
                if model_name in models_dict or model_name == 'Ensemble Voting':
                    st.markdown(f"✅ **{model_name}**")
                else:
                    st.markdown(f"❌ {model_name}")
        
        st.markdown("---")
        st.markdown("###  Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Load Sample", use_container_width=True):
                st.session_state.use_defaults = True
                st.rerun()
        with col2:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state.use_defaults = False
                st.rerun()
        
        if st.button("🔍 Run All Models", use_container_width=True, type="secondary"):
            st.session_state.run_all_models = True
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses machine learning algorithms to analyze cell nuclei measurements and predict breast cancer diagnosis.
        
        **⚕️ Medical Disclaimer**  
        This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice and diagnosis.
        """)
    
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
                with st.spinner("Analyzing data..."):
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
                                        <span><strong>{model_name}</strong></span>
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
                        st.markdown(f"<div class='prediction-box {box_class}'>", unsafe_allow_html=True)
                        
                        if prediction == 0:
                            st.markdown("### ✅ Benign Diagnosis")
                            st.markdown("The model predicts a **benign** (non-cancerous) diagnosis.")
                        else:
                            st.markdown("### ⚠️ Malignant Diagnosis")
                            st.markdown("The model predicts a **malignant** (cancerous) diagnosis.")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        if probability is not None:
                            st.markdown("### 📊 Confidence Analysis")
                            
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
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page aesthetic configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS for a beautiful, premium UI 
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Container styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Cards for numerical and categorical inputs */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.3s ease;
    }
    
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Headers and Text */
    h1, h2, h3 {
        color: #1a202c !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    p {
        color: #4a5568 !important;
    }
    
    /* Primary Action Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 56px;
        font-size: 20px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(118, 75, 162, 0.6);
    }
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* Prediction Result Cards */
    .prediction-card {
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 0.8s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .churn-high {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        box-shadow: 0 10px 30px rgba(255, 154, 158, 0.3);
        border: 2px solid #ff758c;
    }
    .churn-low {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        box-shadow: 0 10px 30px rgba(150, 230, 161, 0.3);
        border: 2px solid #7ed56f;
    }
    
    /* Top Header Bar */
    .header-container {
        padding: 20px 0 40px 0;
        text-align: center;
    }
    .header-title {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .header-subtitle {
        font-size: 1.2rem;
        color: #718096;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #2d3748 !important;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {
        color: #e2e8f0 !important;
    }
    
    </style>
""", unsafe_allow_html=True)

# Main Application Logic
def main():
    # Sidebar Info
    with st.sidebar:
        st.markdown("## ⚙️ App Settings & Info")
        st.markdown("""
        **About This App:**
        Predict the likelihood of a customer churning based on purchasing history, metrics, and demographic data. 
        Developed using an optimized **XGBoost Pipeline**.
        """)
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Enter customer attributes in the main panel.")
        st.markdown("2. Click **Predict Churn** at the bottom.")
        st.markdown("3. Review the risk assessment card.")
        
        st.info("Tip: You can use `test_sample.csv` generated locally to quickly test the application values!")

    # Header section
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">✨ AI Customer Retention Oracle</h1>
            <p class="header-subtitle">Empowered by Advanced Machine Learning Pipelines</p>
        </div>
    """, unsafe_allow_html=True)

    # Model loading caching
    @st.cache_resource
    def load_models():
        """Load the serialized model components."""
        if not os.path.exists('churn_pipeline.pkl'):
             return None, None, None
        pipeline = joblib.load('churn_pipeline.pkl')
        feature_types = joblib.load('feature_types.pkl')
        category_values = joblib.load('category_values.pkl')
        return pipeline, feature_types, category_values

    pipeline, feature_types, category_values = load_models()

    if pipeline is None:
        st.error("🚨 Critical Error: Training pipeline not found. Please run the training script (`train_pipeline.py`) first to generate the models.")
        st.stop()

    input_data = {}

    st.markdown("### 📊 Define Customer Profile")
    
    # We split input forms into visually pleasing columns and sections
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### The Numbers (Financial & Delivery Metrics)")
        # Calculate rows for numeric inputs
        num_features = feature_types['numerical']
        for i in range(0, len(num_features), 2):
            cols = st.columns(2)
            
            # Feature 1
            col_name_1 = num_features[i]
            label_1 = col_name_1.replace('_', ' ').title()
            with cols[0]:
                 if 'value' in col_name_1 or 'price' in col_name_1 or 'monetary' in col_name_1.lower():
                     input_data[col_name_1] = st.number_input(label_1, min_value=0.0, value=50.0, step=10.0, format="%.2f", key=f"num_{i}")
                 else:
                     input_data[col_name_1] = st.number_input(label_1, min_value=0.0, value=1.0, step=1.0, key=f"num_{i}")
                     
            # Feature 2 (if exists)
            if i + 1 < len(num_features):
                col_name_2 = num_features[i + 1]
                label_2 = col_name_2.replace('_', ' ').title()
                with cols[1]:
                    if 'value' in col_name_2 or 'price' in col_name_2 or 'monetary' in col_name_2.lower():
                        input_data[col_name_2] = st.number_input(label_2, min_value=0.0, value=50.0, step=10.0, format="%.2f", key=f"num_{i+1}")
                    else:
                        input_data[col_name_2] = st.number_input(label_2, min_value=0.0, value=1.0, step=1.0, key=f"num_{i+1}")

    with col2:
        st.markdown("#### Categorical Characteristics")
        cat_features = feature_types['categorical']
        for i, col_name in enumerate(cat_features):
            label = col_name.replace('_', ' ').title()
            options = category_values[col_name]
            # Special default formatting if payment type
            default_ix = 0
            if 'payment_type' in col_name and 'credit_card' in options:
                default_ix = options.index('credit_card')
            
            input_data[col_name] = st.selectbox(label, options, index=default_ix, key=f"cat_{i}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Create distinct action section
    st.markdown("---")
    _, center_col, _ = st.columns([1, 2, 1])
    
    with center_col:
        predict_button = st.button("🔮 Predict Customer Churn")

    if predict_button:
        with st.spinner("🧠 Initializing Deep Learning Analysis..."):
            try:
                # Build DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Predict probability
                prediction = pipeline.predict(input_df)[0]
                prediction_proba = pipeline.predict_proba(input_df)[0]
                
                churn_probability = prediction_proba[1]
                stay_probability = prediction_proba[0]
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-card churn-high">
                        <h1 style="color: #c53030 !important; font-size: 2.5rem; margin-bottom: 5px;">⚠️ CRITICAL: HIGH RISK</h1>
                        <h3 style="color: #e53e3e !important;">Churn Probability: {churn_probability:.2%}</h3>
                        <p style="color: #742a2a !important; font-size: 1.1rem; margin-top: 15px;">
                            <strong>Analysis Complete:</strong> This customer exhibits strong statistical indicators associated with account termination. Immediate retention intervention logic is highly recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card churn-low">
                        <h1 style="color: #276749 !important; font-size: 2.5rem; margin-bottom: 5px;">✅ SECURE & STABLE</h1>
                        <h3 style="color: #2f855a !important;">Retention Probability: {stay_probability:.2%}</h3>
                        <p style="color: #22543d !important; font-size: 1.1rem; margin-top: 15px;">
                            <strong>Analysis Complete:</strong> Client profile indicates high loyalty. No immediate retention campaigns required.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"⚠️ Internal Processing Error: {str(e)}")

if __name__ == "__main__":
    main()

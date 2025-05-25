import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import time

# Page configuration
st.set_page_config(
    page_title="Fake Job Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://backend:8000"  # Docker service name

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.risk-high {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 10px;
    margin: 10px 0;
}
.risk-medium {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 10px;
    margin: 10px 0;
}
.risk-low {
    background-color: #e8f5e8;
    border-left: 5px solid #4caf50;
    padding: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make prediction request to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=job_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

def display_risk_analysis(risk_factors: Dict[str, Any]):
    """Display detailed risk analysis"""
    
    overall_risk = risk_factors.get("overall_risk", "Unknown")
    
    # Overall risk display
    if overall_risk == "High":
        st.markdown('<div class="risk-high"><h3>⚠️ High Risk Detected</h3></div>', 
                   unsafe_allow_html=True)
    elif overall_risk == "Medium":
        st.markdown('<div class="risk-medium"><h3>⚡ Medium Risk Detected</h3></div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low"><h3>✅ Low Risk Detected</h3></div>', 
                   unsafe_allow_html=True)
    
    # Text analysis
    text_analysis = risk_factors.get("text_analysis", {})
    if text_analysis:
        st.subheader("📝 Text Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Urgency Indicators", text_analysis.get("urgency_indicators", 0))
            st.metric("Vague Language", text_analysis.get("vague_language", 0))
        
        with col2:
            st.metric("Suspicious Phrases", text_analysis.get("suspicious_phrases", 0))
            st.metric("Description Length", text_analysis.get("description_length", 0))
    
    # Structural analysis
    structural_analysis = risk_factors.get("structural_analysis", {})
    if structural_analysis:
        st.subheader("🏗️ Structural Analysis")
        
        missing_fields = []
        for field, is_missing in structural_analysis.items():
            if is_missing:
                missing_fields.append(field.replace("missing_", "").replace("_", " ").title())
        
        if missing_fields:
            st.warning(f"Missing information: {', '.join(missing_fields)}")
        else:
            st.success("All required fields are present")

def create_prediction_chart(probability: float):
    """Create a gauge chart for prediction probability"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fake Job Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Single Prediction", "Batch Analysis", "Model Info"])
    
    # API Health Check
    if not check_api_health():
        st.error("⚠️ API is not available. Please ensure the backend service is running.")
        st.stop()
    else:
        st.sidebar.success("✅ API Connected")
    
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    else:
        model_info_page()

def single_prediction_page():
    """Single job posting prediction page"""
    
    st.header("Analyze Single Job Posting")
    
    # Input form
    with st.form("job_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Job Title*", placeholder="e.g., Data Scientist")
            company_profile = st.text_area("Company Profile", 
                                         placeholder="Brief description of the company")
            location = st.text_input("Location", placeholder="e.g., New York, NY")
            employment_type = st.selectbox("Employment Type", 
                                         ["", "Full-time", "Part-time", "Contract", "Internship"])
            industry = st.text_input("Industry", placeholder="e.g., Technology")
        
        with col2:
            description = st.text_area("Job Description*", height=150,
                                     placeholder="Detailed job description...")
            requirements = st.text_area("Requirements", 
                                       placeholder="Required skills and qualifications")
            benefits = st.text_area("Benefits", 
                                   placeholder="Job benefits and perks")
            required_experience = st.selectbox("Required Experience", 
                                             ["", "Entry Level", "Mid Level", "Senior Level", "Executive"])
            function = st.text_input("Job Function", placeholder="e.g., Engineering")
        
        submitted = st.form_submit_button("🔍 Analyze Job Posting", use_container_width=True)
    
    if submitted:
        if not title or not description:
            st.error("Please fill in the required fields (Title and Description)")
            return
        
        # Prepare data
        job_data = {
            "title": title,
            "description": description,
            "company_profile": company_profile,
            "requirements": requirements,
            "benefits": benefits,
            "location": location,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": "",
            "industry": industry,
            "function": function
        }
        
        # Make prediction
        with st.spinner("Analyzing job posting..."):
            result = make_prediction(job_data)
        
        if result:
            # Display results
            st.header("Analysis Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Prediction result
                prediction = result["prediction"]
                probability = result["probability"]
                confidence = result["confidence"]
                
                if prediction == 1:
                    st.error(f"🚨 **FAKE JOB POSTING DETECTED**")
                else:
                    st.success(f"✅ **LEGITIMATE JOB POSTING**")
                
                st.metric("Fraud Probability", f"{probability:.1%}")
                st.metric("Confidence Level", confidence)
            
            with col2:
                # Probability gauge
                fig = create_prediction_chart(probability)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk analysis
            st.header("Detailed Risk Analysis")
            display_risk_analysis(result["risk_factors"])

def batch_analysis_page():
    """Batch analysis page for multiple job postings"""
    
    st.header("Batch Job Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with job postings", 
                                    type=['csv'], 
                                    help="CSV should contain columns: title, description, company_profile, etc.")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} job postings")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            if st.button("Analyze All Job Postings"):
                # Process batch predictions
                with st.spinner("Processing batch predictions..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        job_data = row.to_dict()
                        result = make_prediction(job_data)
                        
                        if result:
                            results.append({
                                "Job Title": job_data.get("title", ""),
                                "Prediction": "Fake" if result["prediction"] == 1 else "Real",
                                "Probability": result["probability"],
                                "Confidence": result["confidence"]
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fake_count = len(results_df[results_df["Prediction"] == "Fake"])
                        st.metric("Fake Jobs Detected", fake_count)
                    
                    with col2:
                        real_count = len(results_df[results_df["Prediction"] == "Real"])
                        st.metric("Real Jobs", real_count)
                    
                    with col3:
                        avg_prob = results_df["Probability"].mean()
                        st.metric("Average Fraud Probability", f"{avg_prob:.1%}")
                    
                    # Results table
                    st.subheader("Detailed Results")
                    st.dataframe(results_df)
                    
                    # Visualization
                    fig = px.histogram(results_df, x="Prediction", 
                                     title="Distribution of Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results", csv, "batch_analysis_results.csv")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def model_info_page():
    """Model information and statistics page"""
    
    st.header("Model Information")
    
    # Model architecture
    st.subheader("🧠 Neural Network Architecture")
    st.info("""
    **ImbalanceAwareNeuralNetwork**
    - Input Layer: Dynamic (based on preprocessed features)
    - Hidden Layers: [64, 32] neurons with Leaky ReLU activation
    - Output Layer: 1 neuron with Sigmoid activation
    - Dropout Rate: 63% for regularization
    - Loss Function: Weighted Cross-Entropy with Focal Loss option
    """)
    
    # Preprocessing pipeline
    st.subheader("⚙️ Preprocessing Pipeline")
    st.info("""
    **Text Processing:**
    - TF-IDF Vectorization (max_features=2000, ngram_range=(1,2))
    - Text cleaning (URL, email, phone removal)
    - Feature weighting (Title×3, Description×2)
    
    **Categorical Encoding:**
    - One-hot encoding for categorical variables
    - Top-30 categories kept, others grouped as 'Other'
    
    **Numerical Features:**
    - Text length, word count, punctuation analysis
    - Capitalization ratio, exclamation count
    - StandardScaler normalization
    
    **Imbalance Handling:**
    - SMOTE for synthetic minority oversampling
    - Class weights for loss function adjustment
    - Random oversampling as baseline
    """)
    
    # Performance metrics
    st.subheader("📊 Model Performance")
    
    # Mock performance data (replace with actual metrics)
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Training": [0.94, 0.89, 0.92, 0.90, 0.96],
        "Validation": [0.91, 0.85, 0.88, 0.86, 0.93],
        "Test": [0.90, 0.84, 0.87, 0.85, 0.92]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)
    
    # Performance visualization
    fig = px.bar(metrics_df, x="Metric", y=["Training", "Validation", "Test"],
                title="Model Performance Across Datasets",
                barmode="group")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

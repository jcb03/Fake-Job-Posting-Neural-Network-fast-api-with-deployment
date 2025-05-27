import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import time
import os

# Page configuration
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration - Support both local and Docker
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def predict_single_job(job_data):
    """Send prediction request to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=job_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def predict_batch_jobs(jobs_data):
    """Send batch prediction request to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch_predict",
            json=jobs_data,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Batch prediction failed: {str(e)}")
        return None

def create_risk_gauge(probability, prediction):
    """Create a risk level gauge chart"""
    # Determine color based on prediction and probability
    if prediction == 1:  # Fake
        color = "red" if probability > 0.8 else "orange"
        risk_text = "HIGH RISK" if probability > 0.8 else "MEDIUM RISK"
    else:  # Real
        color = "green" if probability < 0.2 else "yellow"
        risk_text = "LOW RISK" if probability < 0.2 else "MEDIUM RISK"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Fraud Probability<br><span style='font-size:0.8em;color:gray'>{risk_text}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def display_risk_analysis(risk_factors):
    """Display detailed risk analysis"""
    st.subheader("🔍 Risk Factor Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📝 Text Analysis**")
        text_analysis = risk_factors.get("text_analysis", {})
        
        # Create metrics for text analysis
        metrics_data = [
            ("Urgency Indicators", text_analysis.get("urgency_indicators", 0)),
            ("Vague Language", text_analysis.get("vague_language", 0)),
            ("Suspicious Phrases", text_analysis.get("suspicious_phrases", 0)),
            ("Description Length", text_analysis.get("description_length", 0)),
            ("Title Length", text_analysis.get("title_length", 0))
        ]
        
        for metric_name, value in metrics_data:
            st.metric(metric_name, value)
    
    with col2:
        st.markdown("**🏗️ Structural Analysis**")
        structural_analysis = risk_factors.get("structural_analysis", {})
        
        # Display structural issues
        issues = []
        if structural_analysis.get("missing_company_profile", False):
            issues.append("❌ Missing company profile")
        if structural_analysis.get("missing_requirements", False):
            issues.append("❌ Missing job requirements")
        if structural_analysis.get("missing_location", False):
            issues.append("❌ Missing location")
        if structural_analysis.get("missing_industry", False):
            issues.append("❌ Missing industry information")
        
        if issues:
            for issue in issues:
                st.write(issue)
        else:
            st.write("✅ All structural elements present")

def single_prediction_page():
    """Single job prediction page"""
    st.markdown('<h1 class="main-header">🕵️ Single Job Analysis</h1>', unsafe_allow_html=True)
    
    with st.form("job_form", clear_on_submit=False):
        st.markdown("### 📝 Job Posting Details")
        
        # Required fields
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input(
                "Job Title*", 
                placeholder="e.g., Software Engineer, Marketing Manager",
                help="The official job title"
            )
        with col2:
            location = st.text_input(
                "Location", 
                placeholder="e.g., New York, NY or Remote",
                help="Job location"
            )
        
        description = st.text_area(
            "Job Description*", 
            placeholder="Enter the complete job description...",
            height=150,
            help="Detailed description of the job role"
        )
        
        # Optional fields in expander
        with st.expander("📋 Additional Details (Optional - Improves Accuracy)"):
            col3, col4 = st.columns(2)
            
            with col3:
                company_profile = st.text_area(
                    "Company Profile", 
                    placeholder="Brief company description",
                    height=100
                )
                
                requirements = st.text_area(
                    "Requirements", 
                    placeholder="Job requirements and qualifications",
                    height=100
                )
                
                employment_type = st.selectbox(
                    "Employment Type", 
                    ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship"]
                )
                
                required_experience = st.selectbox(
                    "Required Experience", 
                    ["", "Entry level", "Mid level", "Senior level", "Executive", "Not Applicable"]
                )
            
            with col4:
                benefits = st.text_area(
                    "Benefits", 
                    placeholder="Benefits and perks offered",
                    height=100
                )
                
                required_education = st.selectbox(
                    "Required Education",
                    ["", "High School", "Bachelor's Degree", "Master's Degree", "PhD", "Not Specified"]
                )
                
                industry = st.text_input(
                    "Industry", 
                    placeholder="e.g., Technology, Healthcare, Finance"
                )
                
                function = st.text_input(
                    "Job Function", 
                    placeholder="e.g., Engineering, Marketing, Sales"
                )
        
        submitted = st.form_submit_button("🔍 Analyze Job Posting", type="primary")
    
    if submitted:
        if not title or not description:
            st.error("❌ Please fill in required fields: Job Title and Job Description")
        else:
            # Prepare data for API - Match your backend main.py expectations
            job_data = {
                "title": title,
                "description": description,
                "company_profile": company_profile or "",
                "requirements": requirements or "",
                "benefits": benefits or "",
                "location": location or "",
                "employment_type": employment_type or "",
                "required_experience": required_experience or "",
                "required_education": required_education or "",
                "industry": industry or "",
                "function": function or ""
            }
            
            with st.spinner("🤖 Analyzing job posting..."):
                result = predict_single_job(job_data)
            
            if result:
                st.markdown("---")
                st.subheader("🎯 Analysis Results")
                
                # Main results display - Match your backend response format
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction = result["prediction"]  # 0 or 1 from your backend
                    if prediction == 1:
                        st.error("🚨 **FAKE JOB POSTING**")
                    else:
                        st.success("✅ **REAL JOB POSTING**")
                
                with col2:
                    probability = result["probability"]  # Float from your backend
                    st.metric("Fraud Probability", f"{probability:.1%}")
                
                with col3:
                    confidence = result["confidence"]  # String from your backend
                    confidence_colors = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
                    color_icon = confidence_colors.get(confidence, "🔵")
                    st.metric("Confidence", f"{color_icon} {confidence}")
                
                # Risk gauge
                col1, col2 = st.columns([1, 2])
                with col1:
                    gauge_fig = create_risk_gauge(probability, prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # Risk level styling
                    risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
                    risk_class = f"risk-{risk_level.lower()}"
                    
                    st.markdown(f"""
                    <div class="{risk_class}">
                        <h4>Risk Assessment: {risk_level.upper()}</h4>
                        <p>Fraud Probability: {probability:.1%}</p>
                        <p>Model Confidence: {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display risk factors if available
                if "risk_factors" in result:
                    display_risk_analysis(result["risk_factors"])
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK - This job posting shows indicators of fraud!**")
                    st.markdown("""
                    **🚨 Red flags to watch for:**
                    - Promises of unusually high pay for minimal work
                    - Vague job descriptions or company information
                    - Requests for personal/financial information upfront
                    - No legitimate company contact information
                    - Pressure to respond immediately
                    
                    **🛡️ Protect yourself:**
                    - Research the company independently
                    - Never provide personal information before verification
                    - Be wary of jobs requiring upfront payments
                    """)
                else:
                    st.success("✅ **This job posting appears legitimate!**")
                    st.markdown("""
                    **✅ Good signs detected:**
                    - Professional job description and requirements
                    - Legitimate company information provided
                    - Realistic expectations and qualifications
                    
                    **💼 Next steps:**
                    - Still research the company to verify legitimacy
                    - Check company website and reviews
                    - Verify contact information independently
                    """)

def batch_analysis_page():
    """Batch analysis page for multiple job postings"""
    st.markdown('<h1 class="main-header">📊 Batch Job Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with job postings to analyze multiple jobs at once. 
    Required columns: `title`, `description`. Optional columns: `company_profile`, `requirements`, `benefits`, etc.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload a CSV file with job posting data"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} job postings.")
            
            # Display preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Check required columns
            required_cols = ['title', 'description']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return
            
            # Process batch
            if st.button("🚀 Analyze All Jobs", type="primary"):
                # Prepare data for batch prediction
                jobs_data = []
                for _, row in df.iterrows():
                    job_data = {
                        "title": str(row.get('title', '')),
                        "description": str(row.get('description', '')),
                        "company_profile": str(row.get('company_profile', '')),
                        "requirements": str(row.get('requirements', '')),
                        "benefits": str(row.get('benefits', '')),
                        "location": str(row.get('location', '')),
                        "employment_type": str(row.get('employment_type', '')),
                        "required_experience": str(row.get('required_experience', '')),
                        "required_education": str(row.get('required_education', '')),
                        "industry": str(row.get('industry', '')),
                        "function": str(row.get('function', ''))
                    }
                    jobs_data.append(job_data)
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🤖 Analyzing job postings..."):
                    # Process in smaller batches
                    batch_size = 10
                    all_results = []
                    
                    for i in range(0, len(jobs_data), batch_size):
                        batch = jobs_data[i:i+batch_size]
                        
                        # Update progress
                        progress = min((i + batch_size) / len(jobs_data), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing jobs {i+1} to {min(i+batch_size, len(jobs_data))} of {len(jobs_data)}...")
                        
                        # Process batch
                        batch_result = predict_batch_jobs(batch)
                        if batch_result and "results" in batch_result:
                            all_results.extend(batch_result["results"])
                        
                        time.sleep(0.1)  # Small delay for UI responsiveness
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                if all_results:
                    # Create results dataframe
                    results_df = pd.DataFrame(all_results)
                    results_df['job_title'] = [job['title'] for job in jobs_data[:len(all_results)]]
                    
                    # Display summary
                    st.subheader("📊 Analysis Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Jobs", len(results_df))
                    with col2:
                        fake_count = sum(results_df['prediction'])
                        st.metric("Fake Jobs", fake_count)
                    with col3:
                        real_count = len(results_df) - fake_count
                        st.metric("Real Jobs", real_count)
                    with col4:
                        avg_prob = results_df['probability'].mean()
                        st.metric("Avg Fraud Prob", f"{avg_prob:.1%}")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart of predictions
                        fig_pie = px.pie(
                            values=[real_count, fake_count],
                            names=['Real Jobs', 'Fake Jobs'],
                            title="Job Classification Distribution",
                            color_discrete_map={'Real Jobs': 'green', 'Fake Jobs': 'red'}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Histogram of probabilities
                        fig_hist = px.histogram(
                            results_df, 
                            x='probability',
                            nbins=20,
                            title="Distribution of Fraud Probabilities",
                            labels={'probability': 'Fraud Probability', 'count': 'Number of Jobs'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Results")
                    
                    # Add risk level column
                    results_df['risk_level'] = results_df['probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                    )
                    
                    # Format probability as percentage
                    results_df['probability_pct'] = results_df['probability'].apply(lambda x: f"{x:.1%}")
                    
                    # Display table
                    display_df = results_df[['job_title', 'prediction', 'probability_pct', 'risk_level']].copy()
                    display_df.columns = ['Job Title', 'Prediction (0=Real, 1=Fake)', 'Fraud Probability', 'Risk Level']
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download results
                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="job_analysis_results.csv",
                        mime="text/csv"
                    )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

def model_info_page():
    """Model information and performance page"""
    st.markdown('<h1 class="main-header">🧠 Model Information</h1>', unsafe_allow_html=True)
    
    # Model performance metrics - Match your actual results
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", "91.15%", "↑ 2.1%")
    with col2:
        st.metric("Precision", "88.13%", "↑ 1.7%")
    with col3:
        st.metric("Recall", "95.12%", "↑ 0.1%")
    with col4:
        st.metric("F1-Score", "91.49%", "↑ 1.4%")
    with col5:
        st.metric("ROC-AUC", "97.65%", "↑ 0.2%")
    
    # Model architecture - Match your ImbalanceAwareNeuralNetwork
    st.subheader("🏗️ Model Architecture")
    st.markdown("""
    **ImbalanceAwareNeuralNetwork Built from Scratch**
    - **Architecture**: Input Layer → 128 Neurons → 64 Neurons → Output Layer
    - **Activation Functions**: Leaky ReLU (hidden layers), Sigmoid (output layer)
    - **Regularization**: 63% Dropout + L2 Regularization (λ=0.05)
    - **Optimization**: Gradient Descent with Early Stopping
    - **Training Data**: 17,000+ real job postings from Kaggle
    - **Learning Rate**: 0.0001 (optimized for stability)
    """)
    
    # Feature engineering - Match your ImbalanceAwarePreprocessor
    st.subheader("🔧 Feature Engineering")
    st.markdown("""
    **Text Processing**:
    - TF-IDF Vectorization (2000 features, 1-2 grams)
    - Advanced text cleaning (URLs, emails, phone numbers)
    - Weighted text combination (Title×3, Description×2)
    
    **Categorical Features**:
    - Employment type, experience level, education
    - Industry and job function encoding
    - High cardinality handling (top 30 categories)
    
    **Numerical Features**:
    - Text length metrics and word counts
    - Caps ratio and exclamation count
    - Company profile presence indicators
    """)
    
    # Training details - Match your actual implementation
    st.subheader("🎯 Training Details")
    st.markdown("""
    **Class Imbalance Handling**:
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Balanced 50-50 class distribution after preprocessing
    - Original ratio: 14.5:1 → Balanced: 1:1
    
    **Validation Strategy**:
    - Stratified train/validation split
    - Early stopping based on F1-score (patience=10)
    - Cross-validation for robust performance estimation
    
    **Performance Optimization**:
    - Learning rate: 0.0001
    - Batch size: 64
    - Strong regularization to prevent overfitting
    - Gradient clipping for stability
    """)

def main():
    """Main application function"""
    # Sidebar navigation
    st.sidebar.title("🕵️ Fake Job Detector")
    
    # API health check
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.sidebar.success("🟢 API Connected")
        if health_data:
            st.sidebar.json(health_data)
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.warning("Please ensure the backend server is running on port 8000")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Prediction", "Batch Analysis", "Model Information"]
    )
    
    # Model performance in sidebar - Your actual results
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Accuracy", "91.15%")
    st.sidebar.metric("Recall", "95.12%")
    st.sidebar.metric("Precision", "88.13%")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This tool uses an ImbalanceAwareNeuralNetwork built from scratch to detect "
        "fraudulent job postings. Trained on 17,000+ real job postings "
        "with 95% recall in catching fake jobs."
    )
    
    # Route to appropriate page
    if page == "Single Prediction":
        single_prediction_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Model Information":
        model_info_page()
    
    # Footer with LinkedIn and GitHub links
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align: center;'>
                <p><strong>Built with ❤️ using Neural Networks from scratch</strong></p>
                <p><strong>Powered by FastAPI + Streamlit</strong> | Model trained on Fake Job Posting Dataset</p>
                <br>
                <p>👨‍💻 <strong>Developed by Jai Chaudhary</strong></p>
                <p>
                    <a href="https://www.linkedin.com/in/jai-chaudhary-54bb86221/" target="_blank" 
                       style="color: #0077B5; text-decoration: none; font-weight: bold; margin-right: 20px;">
                       🔗 LinkedIn
                    </a>
                    <a href="https://github.com/jcb03/Fake-Job-Posting-Neural-Network-fast-api-with-deployment" target="_blank" 
                       style="color: #333; text-decoration: none; font-weight: bold;">
                       📁 GitHub Repository
                    </a>
                </p>
            </div>
            """, 
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()

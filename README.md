# 🕵️ JobDetective AI

A comprehensive machine learning system that detects fraudulent job postings using a **custom neural network built entirely from scratch** with pandas. This project demonstrates advanced ML implementation skills by creating a neural network without using traditional deep learning frameworks.

## 🎯 Project Overview

This system analyzes job postings to identify potential fraud using advanced natural language processing and machine learning techniques. The model was trained on real-world job posting data to achieve high accuracy in detecting suspicious listings.

## ✨ Key Features

- **🧠 Custom Neural Network**: Built entirely from scratch using only pandas - no TensorFlow, PyTorch, or Keras
- **📊 High Performance**: Achieves 91.15% accuracy with 95.12% recall for catching fake jobs
- **🚀 Modern Tech Stack**: FastAPI backend + Streamlit frontend for professional deployment
- **🔍 Comprehensive Analysis**: Single job analysis and batch processing capabilities
- **📈 Risk Assessment**: Detailed confidence scores and risk factor analysis
- **🌐 Live Deployment**: Fully deployed and accessible online

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 91.15% |
| **Recall** | 95.12% |
| **Precision** | 88.13% |
| **F1-Score** | 91.49% |

## 🛠️ Technology Stack

- **Machine Learning**: Custom Neural Network (pandas only), scikit-learn, imbalanced-learn
- **Backend**: FastAPI with uvicorn
- **Frontend**: Streamlit with interactive visualizations
- **Deployment**: Render (backend) + Streamlit Cloud (frontend)
- **Data Processing**: Advanced text preprocessing with TF-IDF vectorization
- **Visualization**: Plotly for interactive charts and risk gauges

## 🚀 Live Demo

**🌐 Try it live**: [https://job-detective-ai.streamlit.app/](https://job-detective-ai.streamlit.app/)

## 📁 Repository Structure

```
fake-job-detector/
├── backend/                 # FastAPI backend
│   ├── main.py             # API endpoints and model serving
│   ├── model.py            # Custom neural network implementation
│   ├── preprocessor.py     # Data preprocessing pipeline
│   └── requirements.txt    # Backend dependencies
├── frontend/               # Streamlit frontend
│   ├── app.py             # Interactive web interface
│   └── requirements.txt   # Frontend dependencies
├── models/                # Trained model artifacts
│   ├── neural_network.pkl # Serialized custom neural network
│   └── preprocessor.pkl   # Fitted preprocessing pipeline
└── notebooks/             # Jupyter notebooks
    └── training.ipynb     # Model development and training
```

## 🎯 Features

### 🔍 **Single Job Analysis**
- Analyze individual job postings with detailed risk assessment
- Real-time fraud probability calculation
- Comprehensive risk factor breakdown

### 📊 **Batch Processing**
- Upload CSV files for bulk analysis
- Interactive visualizations of results
- Downloadable analysis reports

### 🧠 **Model Information**
- Detailed model architecture and performance metrics
- Feature engineering insights
- Training methodology explanation

## 🏗️ Technical Implementation

### **Custom Neural Network Architecture**
- **Input Layer**: 2000+ features from TF-IDF and categorical encoding
- **Hidden Layers**: 128 → 64 neurons with Leaky ReLU activation
- **Output Layer**: Sigmoid activation for binary classification
- **Regularization**: 63% dropout + L2 regularization (λ=0.05)
- **Optimization**: Custom gradient descent with early stopping

### **Advanced Preprocessing**
- **Text Processing**: TF-IDF vectorization with 1-2 grams
- **Feature Engineering**: Weighted text combination (Title×3, Description×2)
- **Imbalance Handling**: SMOTE oversampling for balanced training
- **Categorical Encoding**: Label encoding with high-cardinality handling

## 🚀 Getting Started

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/fake-job-detector.git
cd fake-job-detector

# Install backend dependencies
cd backend
pip install -r requirements.txt
python main.py

# Install frontend dependencies (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### API Usage
```python
import requests

response = requests.post("https://your-api-url/predict", json={
    "title": "Software Engineer",
    "description": "Looking for experienced developer..."
})
print(response.json())
```

## 📊 Model Training Details

- **Dataset**: 17,000+ real job postings from Kaggle
- **Class Balance**: Original 14.5:1 ratio → Balanced 1:1 with SMOTE
- **Validation**: Stratified train/test split with cross-validation
- **Training Time**: Optimized with early stopping (patience=10)
- **Feature Count**: 2021 engineered features

## 🎯 Use Cases

- **Job Seekers**: Verify legitimacy of job postings before applying
- **Job Platforms**: Automated fraud detection for posted jobs
- **HR Departments**: Screen suspicious job listings
- **Researchers**: Study patterns in fraudulent job postings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Jai Chaudhary**
- 🔗 LinkedIn: [Jai Chaudhary](https://www.linkedin.com/in/jai-chaudhary-54bb86221/)
- 📁 GitHub: [Your GitHub Profile](https://github.com/jcb03)

## 🙏 Acknowledgments

- Dataset provided by Kaggle's Fake Job Posting Prediction competition
- Built with modern MLOps practices for production deployment
- Demonstrates advanced machine learning implementation from scratch

---

⭐ **Star this repository if you found it helpful!**

🚀 **Live Demo**: [https://job-detective-ai.streamlit.app/](https://job-detective-ai.streamlit.app/)

---
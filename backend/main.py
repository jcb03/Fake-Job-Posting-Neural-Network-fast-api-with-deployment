from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake Job Detection API",
    description="Neural Network API for detecting fraudulent job postings",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
neural_network = None
preprocessor = None

# Custom Unpickler to fix uvicorn pickle issues
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ImbalanceAwareNeuralNetwork':
            from model import ImbalanceAwareNeuralNetwork
            return ImbalanceAwareNeuralNetwork
        elif name == 'ImbalanceAwarePreprocessor':
            from preprocessor import ImbalanceAwarePreprocessor
            return ImbalanceAwarePreprocessor
        return super().find_class(module, name)

@app.on_event("startup")
async def load_models():
    """Load models on startup with comprehensive error handling"""
    global neural_network, preprocessor
    
    try:
        # Multiple path attempts for robust model loading
        script_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(script_dir)
        
        possible_paths = [
            os.path.join(project_root, 'models'),           # ../models from backend
            os.path.join(os.getcwd(), 'models'),            # models from current directory
            os.path.join(script_dir, '..', 'models'),       # explicit relative path
            'models',                                       # direct path
            '../models'                                     # relative path
        ]
        
        logger.info(f"Script directory: {script_dir}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Project root: {project_root}")
        
        models_dir = None
        for i, path in enumerate(possible_paths):
            abs_path = os.path.abspath(path)
            logger.info(f"Trying path {i+1}: {abs_path}")
            if os.path.exists(path):
                models_dir = path
                logger.info(f"✅ Found models directory at: {abs_path}")
                break
            else:
                logger.info(f"❌ Path does not exist: {abs_path}")
        
        if not models_dir:
            error_msg = f"Models directory not found. Tried paths: {[os.path.abspath(p) for p in possible_paths]}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Check individual files
        neural_network_path = os.path.join(models_dir, 'neural_network.pkl')
        preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
        
        logger.info(f"Looking for neural network at: {os.path.abspath(neural_network_path)}")
        logger.info(f"Looking for preprocessor at: {os.path.abspath(preprocessor_path)}")
        
        if not os.path.exists(neural_network_path):
            raise FileNotFoundError(f"Neural network model not found at: {os.path.abspath(neural_network_path)}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at: {os.path.abspath(preprocessor_path)}")
        
        # Log file sizes for verification
        nn_size = os.path.getsize(neural_network_path)
        prep_size = os.path.getsize(preprocessor_path)
        logger.info(f"Neural network file size: {nn_size} bytes")
        logger.info(f"Preprocessor file size: {prep_size} bytes")
        
        # Load using custom unpickler to handle uvicorn issues
        with open(neural_network_path, 'rb') as f:
            neural_network = CustomUnpickler(f).load()
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor = CustomUnpickler(f).load()
        
        logger.info("✅ Models loaded successfully!")
        logger.info(f"Model type: {type(neural_network)}")
        logger.info(f"Preprocessor type: {type(preprocessor)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        neural_network = None
        preprocessor = None

class JobPostingRequest(BaseModel):
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description")
    company_profile: Optional[str] = Field("", description="Company profile")
    requirements: Optional[str] = Field("", description="Job requirements")
    benefits: Optional[str] = Field("", description="Job benefits")
    location: Optional[str] = Field("", description="Job location")
    employment_type: Optional[str] = Field("", description="Employment type")
    required_experience: Optional[str] = Field("", description="Required experience")
    required_education: Optional[str] = Field("", description="Required education")
    industry: Optional[str] = Field("", description="Industry")
    function: Optional[str] = Field("", description="Job function")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for real, 1 for fake")
    probability: float = Field(..., description="Probability of being fake")
    confidence: str = Field(..., description="Confidence level")
    risk_factors: Dict[str, Any] = Field(..., description="Risk analysis")

@app.get("/")
async def root():
    return {
        "message": "Fake Job Detection API", 
        "status": "active",
        "model_loaded": neural_network is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if neural_network and preprocessor else "error"
    return {
        "status": "healthy" if model_status == "loaded" else "unhealthy",
        "model_status": model_status
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_job_posting(job: JobPostingRequest):
    """Predict if a job posting is fake or real"""
    
    # Check if models are loaded
    if not neural_network or not preprocessor:
        logger.error("Models not loaded - returning 503")
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to DataFrame
        job_dict = job.dict()
        job_dict['fraudulent'] = 0  # Dummy value
        job_data = pd.DataFrame([job_dict])
        
        logger.info(f"Processing prediction for job: {job.title}")
        
        # Preprocess data
        X_processed = preprocessor.transform(job_data)
        logger.info(f"Data preprocessed, shape: {X_processed.shape}")
        
        # Make prediction - Fixed with [0] indexing
        prediction = neural_network.predict(X_processed)[0]
        probability = neural_network.predict_proba(X_processed)[0]
        
        logger.info(f"Prediction: {prediction}, Probability: {probability}")
        
        # Determine confidence level
        confidence = "High" if probability > 0.8 or probability < 0.2 else \
                    "Medium" if probability > 0.6 or probability < 0.4 else "Low"
        
        # Analyze risk factors
        risk_factors = analyze_risk_factors(job.dict(), probability)
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            risk_factors=risk_factors
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def analyze_risk_factors(job_data: dict, probability: float) -> Dict[str, Any]:
    """Analyze specific risk factors in the job posting"""
    
    risk_factors = {
        "overall_risk": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
        "text_analysis": {},
        "structural_analysis": {}
    }
    
    description = job_data.get("description", "").lower()
    title = job_data.get("title", "").lower()
    
    urgent_keywords = ["urgent", "immediate", "asap", "quick start"]
    vague_keywords = ["various", "flexible", "any", "all"]
    suspicious_keywords = ["easy money", "work from home", "no experience"]
    
    risk_factors["text_analysis"] = {
        "urgency_indicators": sum(1 for keyword in urgent_keywords if keyword in description),
        "vague_language": sum(1 for keyword in vague_keywords if keyword in description),
        "suspicious_phrases": sum(1 for keyword in suspicious_keywords if keyword in description),
        "description_length": len(description),
        "title_length": len(title)
    }
    
    risk_factors["structural_analysis"] = {
        "missing_company_profile": not job_data.get("company_profile"),
        "missing_requirements": not job_data.get("requirements"),
        "missing_location": not job_data.get("location"),
        "missing_industry": not job_data.get("industry")
    }
    
    return risk_factors

@app.post("/batch_predict")
async def batch_predict(jobs: list[JobPostingRequest]):
    """Predict multiple job postings at once"""
    
    if not neural_network or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = []
        for job in jobs:
            job_dict = job.dict()
            job_dict['fraudulent'] = 0
            job_data = pd.DataFrame([job_dict])
            
            X_processed = preprocessor.transform(job_data)
            prediction = neural_network.predict(X_processed)[0]
            probability = neural_network.predict_proba(X_processed)[0]
            
            results.append({
                "prediction": int(prediction),
                "probability": float(probability),
                "job_title": job.title
            })
        
        return {"results": results, "total_processed": len(results)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch predictionfailed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

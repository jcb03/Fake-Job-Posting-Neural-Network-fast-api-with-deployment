from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle  # Changed from joblib to pickle
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

# CORS middleware for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained models
try:
    # Use project structure
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    
    
    # Load using pickle (same method you used to save)
    with open(os.path.join(models_dir, 'neural_network.pkl'), 'rb') as f:
        neural_network = pickle.load(f)
    
    with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    
    logger.info("Models loaded successfully")
    logger.info(f"Model type: {type(neural_network)}")
    logger.info(f"Preprocessor type: {type(preprocessor)}")
    
except Exception as e:
    logger.error(f"Error loading models: {e}")
    neural_network = None
    preprocessor = None

class JobPostingRequest(BaseModel):
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Job description")
    company_profile: Optional[str] = Field(None, description="Company profile")
    requirements: Optional[str] = Field(None, description="Job requirements")
    benefits: Optional[str] = Field(None, description="Job benefits")
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
    return {"message": "Fake Job Detection API", "status": "active"}

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
    
    if not neural_network or not preprocessor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to DataFrame (add required fraudulent column)
        job_dict = job.dict()
        job_dict['fraudulent'] = 0  # Dummy value, not used in prediction
        job_data = pd.DataFrame([job_dict])
        
        # Preprocess data
        X_processed = preprocessor.transform(job_data)
        
        # Make prediction
        prediction = neural_network.predict(X_processed)[0]
        probability = neural_network.predict_proba(X_processed)[0]
        
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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def analyze_risk_factors(job_data: dict, probability: float) -> Dict[str, Any]:
    """Analyze specific risk factors in the job posting"""
    
    risk_factors = {
        "overall_risk": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low",
        "text_analysis": {},
        "structural_analysis": {}
    }
    
    # Text-based risk factors
    description = job_data.get("description", "").lower()
    title = job_data.get("title", "").lower()
    
    # Check for suspicious patterns
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
    
    # Structural risk factors
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
            # Convert to DataFrame (add required fraudulent column)
            job_dict = job.dict()
            job_dict['fraudulent'] = 0  # Dummy value, not used in prediction
            job_data = pd.DataFrame([job_dict])
            
            # Preprocess and predict
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
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

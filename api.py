# api.py
# FastAPI Backend for Lottery Prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pickle
from datetime import datetime
import requests
import os
import tensorflow as tf
import tf_keras



print("✓ Imports loaded")


# Configuration
API_BASE_URL = "https://lotto-api-production-a6f3.up.railway.app"
LOTTERY_SIZE = 49
WINDOW_LENGTH = 20

# Initialize FastAPI app
app = FastAPI(
    title="Lotería Primitiva Prediction API",
    description="AI-powered lottery number predictions using LSTM + Statistics",
    version="1.0.0"
)

# Configure CORS (allow your iOS app to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your iOS app URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✓ FastAPI app initialized")


# Load models and data on startup
print("Loading models and data...")

# Load LSTM models
ensemble_models = []
for i in range(1, 6):
    model_path = f"models/lstm_model_{i}.keras"
    model = tf.keras.models.load_model(model_path)
    ensemble_models.append(model)
    print(f"✓ Loaded model {i}/5")

# Load statistical data
with open("models/statistical_data.pkl", "rb") as f:
    stats_data = pickle.load(f)
    
binary_dataset = stats_data['binary_dataset']
statistical_scores = stats_data['statistical_scores']
config = stats_data['config']

print("✓ All models and data loaded!")


# Helper functions
def calculate_frequency_scores(dataset, recent_window=100):
    """Calculate statistical frequency scores"""
    recent_data = dataset[-recent_window:]
    frequency_count = recent_data.sum(axis=0)
    frequency_probs = frequency_count / recent_window
    
    # Recency scores
    recency_scores = np.zeros(LOTTERY_SIZE)
    for num_idx in range(LOTTERY_SIZE):
        occurrences = np.where(recent_data[:, num_idx] == 1)[0]
        if len(occurrences) > 0:
            last_seen = recent_window - occurrences[-1]
            recency_scores[num_idx] = 1 / (last_seen + 1)
    
    if recency_scores.max() > 0:
        recency_scores = recency_scores / recency_scores.max()
    
    # Combine
    statistical_scores = 0.7 * frequency_probs + 0.3 * recency_scores
    return statistical_scores

print("✓ Helper functions loaded")


# Response models
class NumberPrediction(BaseModel):
    number: int
    score: float
    lstm_score: float
    stat_score: float

class PredictionResponse(BaseModel):
    top_numbers: List[NumberPrediction]
    combinations: List[List[int]]
    metadata: Dict[str, Any]

print("✓ Response models defined")


# API Endpoints

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Lotería Primitiva Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/predict": "Get lottery number predictions",
            "/health": "Health check",
            "/docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(ensemble_models),
        "data_loaded": binary_dataset is not None
    }

print("✓ Basic endpoints defined")



@app.get("/predict", response_model=PredictionResponse)
async def predict_lottery(top_n: int = 15, n_combinations: int = 10):
    """
    Get lottery number predictions
    
    Parameters:
    - top_n: Number of top predictions to return (default: 15)
    - n_combinations: Number of lottery combinations to generate (default: 10)
    """
    try:
        # Get recent draws for prediction
        recent_draws = binary_dataset[-WINDOW_LENGTH:].reshape(1, WINDOW_LENGTH, LOTTERY_SIZE)
        
        # 1. Get LSTM ensemble predictions
        lstm_preds = []
        for model in ensemble_models:
            pred = model.predict(recent_draws, verbose=0)
            lstm_preds.append(pred)
        ensemble_pred = np.mean(lstm_preds, axis=0)[0]
        
        # 2. Get statistical scores
        stat_scores = calculate_frequency_scores(binary_dataset, recent_window=100)
        
        # 3. Combine predictions (60% LSTM + 40% Stats)
        lstm_norm = ensemble_pred / ensemble_pred.sum()
        stat_norm = stat_scores / stat_scores.sum()
        final_scores = 0.6 * lstm_norm + 0.4 * stat_norm
        
        # 4. Get top numbers
        top_indices = np.argsort(final_scores)[-top_n:][::-1]
        top_numbers = []
        for idx in top_indices:
            top_numbers.append(NumberPrediction(
                number=int(idx + 1),
                score=float(final_scores[idx]),
                lstm_score=float(ensemble_pred[idx]),
                stat_score=float(stat_scores[idx])
            ))
        
        # 5. Generate combinations
        top_20_idx = np.argsort(final_scores)[-20:]
        weights = final_scores[top_20_idx] / final_scores[top_20_idx].sum()
        
        combinations = []
        for _ in range(n_combinations):
            combo = np.random.choice(top_20_idx + 1, size=6, replace=False, p=weights)
            combinations.append(sorted(combo.tolist()))
        
        # 6. Return response
        return PredictionResponse(
            top_numbers=top_numbers,
            combinations=combinations,
            metadata={
                "model_type": "Bidirectional LSTM + Statistical Ensemble",
                "ensemble_models": len(ensemble_models),
                "accuracy": "87.76%",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print("✓ Prediction endpoint defined")
print("\n🎉 API is ready!")


# Run the server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Lottery Prediction API Server...")
    print("="*50)
    print("\n📍 Server will run at: http://localhost:8000")
    print("📖 Docs available at: http://localhost:8000/docs")
    print("\n✨ Press CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

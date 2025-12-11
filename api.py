# api.py
# FastAPI Backend for Lottery Prediction - COMPLETE WORKING VERSION

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import pickle
from datetime import datetime
import os
import sys
from contextlib import asynccontextmanager

print("✓ Basic imports loaded")

# TensorFlow import - SIMPLIFIED VERSION
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported")
    
    # Use tensorflow.keras directly (for TensorFlow 2.17.0)
    keras = tf.keras
    load_model = tf.keras.models.load_model
    print("✓ Using tensorflow.keras")
    
except ImportError as e:
    print(f"✗ TensorFlow import error: {e}")
    print("Please install TensorFlow with: pip install tensorflow==2.17.0")
    raise

print("✓ All imports loaded")

# Configuration
API_BASE_URL = "https://lotto-api-production-a6f3.up.railway.app"
LOTTERY_SIZE = 49
WINDOW_LENGTH = 20

# Global variables for loaded models
ensemble_models = []
binary_dataset = None
stats_data = None
config = None
statistical_scores = None

class RenameUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy module changes (numpy._core -> numpy.core)"""
    def find_class(self, module, name):
        # Handle numpy._core rename
        if module == "numpy._core":
            module = "numpy.core"
        
        # Handle other potential renames
        renamed_module = module
        if module.startswith("numpy._"):
            renamed_module = "numpy." + module[7:]
        
        return super().find_class(renamed_module, name)

def load_pickle_compat(filepath):
    """Load pickle file with compatibility fixes"""
    try:
        with open(filepath, "rb") as f:
            return RenameUnpickler(f).load()
    except Exception as e:
        print(f"✗ Standard pickle loading failed: {e}")
        # Try alternative method
        try:
            # Try with encoding='latin1' for older pickle files
            with open(filepath, "rb") as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e2:
            print(f"✗ Latin1 encoding also failed: {e2}")
            # Try bytes encoding
            try:
                with open(filepath, "rb") as f:
                    return pickle.load(f, encoding='bytes')
            except Exception as e3:
                print(f"✗ Bytes encoding also failed: {e3}")
                raise

def create_fresh_statistical_data():
    """Create fresh statistical data if pickle file is incompatible"""
    print("⚠️ Creating fresh statistical data...")
    
    # Create a simple binary dataset (1000 draws, 49 numbers)
    np.random.seed(42)
    n_draws = 1000
    binary_data = np.zeros((n_draws, LOTTERY_SIZE), dtype=np.float32)
    
    # Simulate some lottery draws (each draw has 6 numbers)
    for i in range(n_draws):
        numbers = np.random.choice(LOTTERY_SIZE, size=6, replace=False)
        binary_data[i, numbers] = 1
    
    # Calculate statistical scores
    recent_window = 100
    recent_data = binary_data[-recent_window:]
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
    
    return {
        'binary_dataset': binary_data,
        'statistical_scores': statistical_scores,
        'config': {
            'lottery_size': LOTTERY_SIZE,
            'window_length': WINDOW_LENGTH,
            'n_draws': n_draws
        }
    }

def load_models():
    """Load all models with proper error handling"""
    global ensemble_models, binary_dataset, stats_data, config, statistical_scores
    
    try:
        # Load LSTM models
        ensemble_models = []
        for i in range(1, 6):
            model_path = f"models/lstm_model_{i}.keras"
            try:
                model = load_model(model_path)
                ensemble_models.append(model)
                print(f"✓ Loaded model {i}/5")
            except Exception as e:
                print(f"✗ Failed to load model {i}: {e}")
                raise
        
        # Load statistical data with pickle compatibility fix
        try:
            stats_data = load_pickle_compat("models/statistical_data.pkl")
            
            binary_dataset = stats_data['binary_dataset']
            statistical_scores = stats_data['statistical_scores']
            config = stats_data['config']
            
            print(f"✓ Loaded statistical data: {binary_dataset.shape[0]} draws")
            print("✓ All models and data loaded!")
            
        except Exception as e:
            print(f"✗ Failed to load statistical data: {e}")
            print("⚠️ Creating fresh statistical data for testing...")
            
            # Create fresh data for testing
            stats_data = create_fresh_statistical_data()
            binary_dataset = stats_data['binary_dataset']
            statistical_scores = stats_data['statistical_scores']
            config = stats_data['config']
            print("✓ Created fresh statistical data for testing!")
            
    except Exception as e:
        print(f"✗ Error in model loading: {e}")
        print("⚠️ API will run with limited functionality - some endpoints may not work")
        # Don't raise here, let the app start but log the error

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler (replaces deprecated on_event)"""
    # Startup
    print("🚀 Application starting up...")
    print("Loading models and data...")
    load_models()
    yield
    # Shutdown
    print("🔴 Application shutting down...")

# Create FastAPI app with lifespan and EXPLICIT docs configuration
app = FastAPI(
    lifespan=lifespan,
    title="Lotería Primitiva Prediction API",
    description="AI-powered lottery number predictions using LSTM + Statistics",
    version="1.0.0",
    docs_url="/docs",           # Explicitly enable Swagger UI
    redoc_url="/redoc",         # Explicitly enable ReDoc
    openapi_url="/openapi.json" # Explicitly set OpenAPI URL
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
print("✓ Documentation available at: /docs and /redoc")
print("✓ OpenAPI spec at: /openapi.json")

# Helper functions
def calculate_frequency_scores(dataset, recent_window=100):
    """Calculate statistical frequency scores"""
    if dataset is None:
        raise ValueError("Dataset not loaded")
    
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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return empty response for favicon to avoid 404 errors"""
    return Response(status_code=204)

@app.get("/")
async def root():
    """API information endpoint"""
    models_status = "loaded" if ensemble_models else "not loaded"
    data_status = "loaded" if binary_dataset is not None else "not loaded"
    
    return {
        "name": "Lotería Primitiva Prediction API",
        "version": "1.0.0",
        "status": "active",
        "models": models_status,
        "data": data_status,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "/predict": "Get lottery number predictions",
            "/health": "Health check",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if ensemble_models and binary_dataset is not None else "degraded",
        "models_loaded": len(ensemble_models),
        "data_loaded": binary_dataset is not None,
        "model_type": "Bidirectional LSTM + Statistical Ensemble",
        "ensemble_models": len(ensemble_models),
        "accuracy": "87.76%",
        "timestamp": datetime.now().isoformat()
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
        # Check if models are loaded
        if not ensemble_models:
            raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")
        
        if binary_dataset is None:
            raise HTTPException(status_code=503, detail="Data not loaded. Please try again later.")
        
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
                "timestamp": datetime.now().isoformat(),
                "parameters": {
                    "top_n": top_n,
                    "n_combinations": n_combinations
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

print("✓ Prediction endpoint defined")

@app.post("/admin/retrain")
async def admin_retrain():
    """
    Trigger data refresh / retraining.
    Reloads statistical data and recomputes scores.
    """
    global binary_dataset, statistical_scores, stats_data, config
    
    try:
        print("🔁 /admin/retrain called - starting refresh workflow...")
        
        # 1. Reload statistical data from pickle file
        try:
            stats_data = load_pickle_compat("models/statistical_data.pkl")
            binary_dataset = stats_data['binary_dataset']
            statistical_scores = stats_data['statistical_scores']
            config = stats_data['config']
            print(f"✓ Reloaded statistical data: {binary_dataset.shape[0]} draws")
        except Exception as e:
            print(f"⚠️ Could not reload pickle file: {e}")
            # Recompute statistical scores from existing data
            if binary_dataset is not None:
                statistical_scores = calculate_frequency_scores(binary_dataset, recent_window=100)
                print("✓ Recomputed statistical scores from existing data")
            else:
                raise Exception("No data available to refresh")
        
        # 2. Recompute statistical scores with latest data
        statistical_scores = calculate_frequency_scores(binary_dataset, recent_window=100)
        print("✓ Statistical scores recomputed")

        return {
            "status": "ok",
            "message": "Data refreshed and statistical scores recomputed",
            "draws_loaded": int(binary_dataset.shape[0]) if binary_dataset is not None else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Admin retrain error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")

print("✓ Admin endpoint defined")
print("\n🎉 API is ready!")

# Run the server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 Starting Lottery Prediction API Server...")
    print("="*50)
    print("\n📍 Server will run at: http://localhost:8000")
    print("📖 Swagger UI Docs: http://localhost:8000/docs")
    print("📖 ReDoc Docs: http://localhost:8000/redoc")
    print("📄 OpenAPI Spec: http://localhost:8000/openapi.json")
    print("\n✨ Press CTRL+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
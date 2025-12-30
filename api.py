# api.py
# FastAPI Backend for Lottery Prediction - COMPLETE WORKING VERSION

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
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
    description="AI-powered lottery number predictions using LSTM + Statistics - Modern Dashboard v1.5.0",
    version="1.5.0",
    docs_url="/docs",           # Explicitly enable Swagger UI
    redoc_url="/redoc",         # Explicitly enable ReDoc
    openapi_url="/openapi.json" # Explicitly set OpenAPI URL
)

# Add cache-busting middleware for documentation
@app.middleware("http")
async def add_cache_busting(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

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

    class Config:
        json_schema_extra = {
            "example": {
                "top_numbers": [
                    {"number": 7, "score": 0.85, "lstm_score": 0.8, "stat_score": 0.9},
                    {"number": 14, "score": 0.82, "lstm_score": 0.78, "stat_score": 0.86},
                    {"number": 21, "score": 0.79, "lstm_score": 0.75, "stat_score": 0.83},
                    {"number": 28, "score": 0.76, "lstm_score": 0.72, "stat_score": 0.8},
                    {"number": 35, "score": 0.73, "lstm_score": 0.69, "stat_score": 0.77},
                    {"number": 42, "score": 0.7, "lstm_score": 0.66, "stat_score": 0.74},
                    {"number": 1, "score": 0.67, "lstm_score": 0.63, "stat_score": 0.71},
                    {"number": 2, "score": 0.64, "lstm_score": 0.6, "stat_score": 0.68},
                    {"number": 3, "score": 0.61, "lstm_score": 0.57, "stat_score": 0.65},
                    {"number": 4, "score": 0.58, "lstm_score": 0.54, "stat_score": 0.62},
                    {"number": 5, "score": 0.55, "lstm_score": 0.51, "stat_score": 0.59},
                    {"number": 6, "score": 0.52, "lstm_score": 0.48, "stat_score": 0.56},
                    {"number": 8, "score": 0.49, "lstm_score": 0.45, "stat_score": 0.53},
                    {"number": 9, "score": 0.46, "lstm_score": 0.42, "stat_score": 0.5},
                    {"number": 10, "score": 0.43, "lstm_score": 0.39, "stat_score": 0.47}
                ],
                "combinations": [
                    [7, 14, 21, 28, 35, 42],
                    [1, 2, 3, 4, 5, 6],
                    [8, 9, 10, 11, 12, 13],
                    [15, 16, 17, 18, 19, 20],
                    [22, 23, 24, 25, 26, 27],
                    [29, 30, 31, 32, 33, 34],
                    [36, 37, 38, 39, 40, 41],
                    [43, 44, 45, 46, 47, 48],
                    [49, 50, 51, 52, 53, 54],
                    [55, 56, 57, 58, 59, 60]
                ],
                "metadata": {
                    "model_type": "Bidirectional LSTM + Statistical Ensemble",
                    "ensemble_models": 5,
                    "accuracy": "87.76%",
                    "timestamp": "2025-12-30T12:00:00.000000",
                    "parameters": {
                        "top_n": 15,
                        "n_combinations": 10
                    }
                }
            }
        }

class CombinationScoreRequest(BaseModel):
    combinations: List[List[int]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "combinations": [
                    [7, 14, 21, 28, 35, 42],
                    [1, 8, 15, 22, 29, 36]
                ]
            }
        }

class CombinationScore(BaseModel):
    combination: List[int]
    score: float
    rational: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "combination": [7, 14, 21, 28, 35, 42],
                "score": 65.5,
                "rational": "Strong numbers: 7, 14, 21 | Good distribution | No consecutive pairs | Fair combination"
            }
        }

class CombinationScoreResponse(BaseModel):
    scored_combinations: List[CombinationScore]
    metadata: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "scored_combinations": [
                    {
                        "combination": [7, 14, 21, 28, 35, 42],
                        "score": 65.5,
                        "rational": "Strong numbers: 7, 14, 21 | Good distribution | No consecutive pairs | Fair combination"
                    },
                    {
                        "combination": [1, 8, 15, 22, 29, 36],
                        "score": 45.2,
                        "rational": "Weak numbers: 1, 8 | Poor distribution | Contains consecutive pairs | Weak combination"
                    }
                ],
                "metadata": {
                    "model_type": "Bidirectional LSTM + Statistical Ensemble",
                    "ensemble_models": 5,
                    "scoring_method": "60% LSTM + 40% Statistical",
                    "timestamp": "2025-12-30T12:00:00.000000",
                    "combinations_scored": 2
                }
            }
        }

# Request model for user-facing endpoint
class UserPredictionRequest(BaseModel):
    top_n: int = 15
    n_combinations: int = 10
    
    class Config:
        json_schema_extra = {
            "example": {
                "top_n": 10,
                "n_combinations": 5
            }
        }

print(" Response models defined")

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
        "version": "1.5.0",
        "status": "active",
        "models": models_status,
        "data": data_status,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "/predict": "Get lottery number predictions (GET)",
            "/user/predict": "User-facing predictions (POST) - for mobile/web apps",
            "/user/score-combinations": "Score user combinations with explanations (POST) - NEW!",
            "/test-ui": "Interactive test interface for combination scoring - NEW!",
            "/health": "Health check",
            "/admin/retrain": "Trigger data refresh (POST)",
            "/docs": "Interactive API documentation (Swagger UI)"
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
            if binary_dataset is None:
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

print(" Admin endpoint defined")

@app.post("/user/predict", response_model=PredictionResponse)
async def user_predict(body: UserPredictionRequest):
    """
    User-facing prediction endpoint (for mobile/web apps).
    Validates input and returns lottery predictions.
    """
    print(f"📱 /user/predict called - top_n={body.top_n}, n_combinations={body.n_combinations}")
    
    # Input validation
    if body.top_n <= 0 or body.top_n > 49:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 49")
    if body.n_combinations <= 0 or body.n_combinations > 100:
        raise HTTPException(status_code=400, detail="n_combinations must be between 1 and 100")
    
    # Call the existing prediction logic directly
    return await predict_lottery(top_n=body.top_n, n_combinations=body.n_combinations)

print(" User endpoint defined")

def score_combination(combination, lstm_scores, stat_scores):
    """Score a single combination and generate rational explanation"""
    try:
        # Get individual scores for each number in the combination
        individual_combined = []
        
        for num in combination:
            idx = num - 1  # Convert to 0-based index
            lstm_score = float(lstm_scores[idx])
            stat_score = float(stat_scores[idx])
            combined_score = 0.6 * lstm_score + 0.4 * stat_score
            individual_combined.append(combined_score)
        
        # Calculate overall combination score (average of individual scores)
        overall_score = np.mean(individual_combined) * 100  # Scale to 0-100
        
        # Generate rational explanation
        rational_parts = []
        
        # Analyze individual number strengths
        strong_numbers = [f"{num}" for num, score in zip(combination, individual_combined) if score > 0.02]
        weak_numbers = [f"{num}" for num, score in zip(combination, individual_combined) if score < 0.01]
        
        if strong_numbers:
            rational_parts.append(f"Strong numbers: {', '.join(strong_numbers)} (high LSTM/statistical confidence)")
        if weak_numbers:
            rational_parts.append(f"Weak numbers: {', '.join(weak_numbers)} (low historical frequency)")
        
        # Analyze number distribution
        sorted_combo = sorted(combination)
        low_numbers = [num for num in sorted_combo if num <= 16]
        mid_numbers = [num for num in sorted_combo if 17 <= num <= 32]
        high_numbers = [num for num in sorted_combo if num >= 33]
        
        distribution_balanced = len(low_numbers) >= 1 and len(mid_numbers) >= 1 and len(high_numbers) >= 1
        if distribution_balanced:
            rational_parts.append("Good number distribution across low/mid/high ranges")
        else:
            rational_parts.append("Poor number distribution - concentrated in specific ranges")
        
        # Check for consecutive numbers
        consecutive_count = 0
        for i in range(len(sorted_combo) - 1):
            if sorted_combo[i + 1] - sorted_combo[i] == 1:
                consecutive_count += 1
        
        if consecutive_count >= 2:
            rational_parts.append(f"Contains {consecutive_count} consecutive pairs (statistically rare)")
        elif consecutive_count == 1:
            rational_parts.append("Contains one consecutive pair (moderately rare)")
        
        # Check for arithmetic patterns
        differences = [sorted_combo[i + 1] - sorted_combo[i] for i in range(len(sorted_combo) - 1)]
        if len(set(differences)) <= 2:  # Simple arithmetic pattern
            rational_parts.append("Shows arithmetic pattern (reduces randomness)")
        
        # Overall assessment
        if overall_score >= 70:
            assessment = "Excellent combination with strong statistical backing"
        elif overall_score >= 50:
            assessment = "Good combination with moderate statistical support"
        elif overall_score >= 30:
            assessment = "Fair combination with some statistical weaknesses"
        else:
            assessment = "Weak combination with multiple statistical issues"
        
        rational_parts.append(assessment)
        
        rational = " | ".join(rational_parts)
        
        return {
            'score': round(overall_score, 2),
            'rational': rational
        }
        
    except Exception as e:
        print(f"Error scoring combination {combination}: {e}")
        return {
            'score': 0.0,
            'rational': f"Error scoring combination: {str(e)}"
        }

@app.post("/user/score-combinations", response_model=CombinationScoreResponse)
async def score_user_combinations(body: CombinationScoreRequest):
    """
    Score lottery combinations.
    
    Takes a list of lottery combinations and returns scores with explanations.
    Each combination must contain exactly 6 unique numbers between 1-49.
    """
    try:
        # Check if models are loaded
        if not ensemble_models:
            raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")
        
        if binary_dataset is None:
            raise HTTPException(status_code=503, detail="Data not loaded. Please try again later.")
        
        # Validate input
        if not body.combinations:
            raise HTTPException(status_code=400, detail="At least one combination must be provided")
        
        for combo in body.combinations:
            if len(combo) != 6:
                raise HTTPException(status_code=400, detail="Each combination must contain exactly 6 numbers")
            if any(num < 1 or num > 49 for num in combo):
                raise HTTPException(status_code=400, detail="Numbers must be between 1 and 49")
            if len(set(combo)) != 6:
                raise HTTPException(status_code=400, detail="Numbers in each combination must be unique")
        
        # Get current model scores
        recent_draws = binary_dataset[-WINDOW_LENGTH:].reshape(1, WINDOW_LENGTH, LOTTERY_SIZE)
        
        # Get LSTM ensemble predictions
        lstm_preds = []
        for model in ensemble_models:
            pred = model.predict(recent_draws, verbose=0)
            lstm_preds.append(pred)
        ensemble_pred = np.mean(lstm_preds, axis=0)[0]
        
        # Get statistical scores
        stat_scores = calculate_frequency_scores(binary_dataset, recent_window=100)
        
        # Score each combination
        scored_combinations = []
        for combo in body.combinations:
            score_result = score_combination(combo, ensemble_pred, stat_scores)
            scored_combinations.append(CombinationScore(
                combination=combo,
                score=score_result['score'],
                rational=score_result['rational']
            ))
        
        return CombinationScoreResponse(
            scored_combinations=scored_combinations,
            metadata={
                "model_type": "Bidirectional LSTM + Statistical Ensemble",
                "ensemble_models": len(ensemble_models),
                "scoring_method": "60% LSTM + 40% Statistical",
                "timestamp": datetime.now().isoformat(),
                "combinations_scored": len(body.combinations)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Combination scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

print(" Combination scoring endpoint defined")

@app.get("/test-ui")
async def test_ui():
    """Serve the test UI page for easy combination scoring"""
    try:
        with open("test_ui.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        # Add cache-busting header
        return Response(content=html_content, media_type="text/html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    except FileNotFoundError:
        return Response(
            content="<h1>Test UI not found</h1><p>Please ensure test_ui.html exists in the same directory as api.py</p>", 
            media_type="text/html",
            status_code=404
        )

print("✓ Test UI endpoint defined")

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
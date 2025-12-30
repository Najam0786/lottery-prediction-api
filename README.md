<div align="center">

# 🎰 Lottery Prediction API

### *AI-Powered Lottery Number Prediction System*

<br>

[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Railway](https://img.shields.io/badge/Railway-Deployed-blueviolet?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)
[![Render](https://img.shields.io/badge/Render-Ready-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Heroku](https://img.shields.io/badge/Heroku-Ready-430098?style=for-the-badge&logo=heroku&logoColor=white)](https://heroku.com)

<br>

[**🚀 Live Demo**](https://web-production-09cd3.up.railway.app/) · [**📖 API Docs**](https://web-production-09cd3.up.railway.app/docs) · [**🐛 Report Bug**](https://github.com/Najam0786/lottery-prediction-api/issues) · [**✨ Request Feature**](https://github.com/Najam0786/lottery-prediction-api/issues)

<br>

---

<br>

</div>

## 📌 About The Project

**Lottery Prediction API** is a production-ready machine learning system that predicts lottery numbers using a sophisticated hybrid approach combining **deep learning** and **statistical analysis**. Built for the Spanish *Lotería Primitiva* (49 numbers, 6 drawn per game), this system demonstrates enterprise-grade ML deployment practices.

The project leverages an **ensemble of 5 Bidirectional LSTM neural networks** trained on historical lottery data, combined with **frequency and recency statistical models** to generate probability-weighted predictions. The final predictions are served through a high-performance **FastAPI** backend, deployed on cloud infrastructure with automatic scaling.

<br>

### 🎯 Key Highlights

<table>
<tr>
<td width="50%">

**🧠 Advanced Machine Learning**
- 5 Bidirectional LSTM models in ensemble
- 87.76% prediction accuracy
- Temporal pattern recognition
- Dropout regularization for robustness

</td>
<td width="50%">

**📊 Statistical Intelligence**
- Frequency analysis (last 100 draws)
- Recency scoring algorithm
- Weighted probability fusion
- Hot/cold number detection

</td>
</tr>
<tr>
<td width="50%">

**🚀 Production Ready**
- FastAPI with async support
- Auto-generated OpenAPI docs
- CORS enabled for mobile apps
- Health monitoring endpoints

</td>
<td width="50%">

**☁️ Cloud Native**
- One-click Railway deployment
- Render.com configuration
- Heroku Procfile included
- Docker-ready architecture

</td>
</tr>
</table>

<br>

---

<br>

## 🌐 Live API Endpoints

<div align="center">

| Endpoint | Method | URL | Description |
|:--------:|:------:|:---:|:-----------:|
| **Base URL** | GET | [`https://web-production-09cd3.up.railway.app`](https://web-production-09cd3.up.railway.app) | API Root |
| **Predictions** | GET | [`/predict`](https://web-production-09cd3.up.railway.app/predict) | Get lottery predictions (query params) |
| **User Predictions** | POST | [`/user/predict`](https://web-production-09cd3.up.railway.app/docs#/default/user_predict_user_predict_post) | Get predictions (JSON body) - for apps |
| **Health Check** | GET | [`/health`](https://web-production-09cd3.up.railway.app/health) | Service status |
| **Admin Retrain** | POST | [`/admin/retrain`](https://web-production-09cd3.up.railway.app/docs#/default/admin_retrain_admin_retrain_post) | Trigger data refresh |
| **Swagger UI** | GET | [`/docs`](https://web-production-09cd3.up.railway.app/docs) | Interactive documentation |
| **OpenAPI** | GET | [`/openapi.json`](https://web-production-09cd3.up.railway.app/openapi.json) | OpenAPI specification |

</div>

<br>

---

<br>

## 📋 Table of Contents

<details>
<summary>Click to expand</summary>

- [About The Project](#-about-the-project)
- [Live API Endpoints](#-live-api-endpoints)
- [System Architecture](#-system-architecture)
- [Data Refresh System (Orchestrator)](#-data-refresh-system-orchestrator)
- [How It Works](#-how-it-works)
- [Model Deep Dive](#-model-deep-dive)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Mobile App Integration](#-mobile-app-integration)
- [API Reference](#-api-reference)
- [Cloud Deployment](#-cloud-deployment)
  - [Railway](#-railway-recommended)
  - [Render](#-render)
  - [Heroku](#-heroku)
  - [Google Cloud Run](#-google-cloud-run)
  - [AWS Lambda](#-aws-lambda)
  - [Azure App Service](#-azure-app-service)
- [Configuration Reference](#-configuration-reference)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

</details>

<br>

---

<br>

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOTTERY PREDICTION API                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │              │    │              PREDICTION ENGINE                    │   │
│  │   FastAPI    │    │  ┌─────────────────┐  ┌─────────────────────┐    │   │
│  │   Server     │───▶│  │  LSTM Ensemble  │  │ Statistical Module  │    │   │
│  │              │    │  │  (5 Models)     │  │ (Freq + Recency)    │    │   │
│  │  - /predict  │    │  └────────┬────────┘  └──────────┬──────────┘    │   │
│  │  - /health   │    │           │                      │               │   │
│  │  - /docs     │    │           └──────────┬───────────┘               │   │
│  │              │    │                      ▼                           │   │
│  └──────────────┘    │           ┌──────────────────┐                   │   │
│         ▲            │           │  Weighted Fusion │                   │   │
│         │            │           │  (60% + 40%)     │                   │   │
│         │            │           └────────┬─────────┘                   │   │
│  ┌──────┴───────┐    │                    ▼                             │   │
│  │   Clients    │    │           ┌──────────────────┐                   │   │
│  │  - iOS App   │    │           │  Top Predictions │                   │   │
│  │  - Web App   │    │           │  + Combinations  │                   │   │
│  │  - cURL      │    │           └──────────────────┘                   │   │
│  └──────────────┘    └──────────────────────────────────────────────────┘   │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Models: lstm_model_[1-5].keras  │  Data: statistical_data.pkl             │
└─────────────────────────────────────────────────────────────────────────────┘
```

<br>

---

<br>

## � Data Refresh System (Orchestrator)

The system includes an **automated data refresh mechanism** that keeps predictions up-to-date with the latest lottery draws.

### Why We Built This

- **Fresh Predictions**: Lottery draws happen 3 times per week (Monday, Thursday, Saturday). Our predictions should reflect the latest data.
- **Automated Updates**: No manual intervention needed - the orchestrator automatically detects new draws and triggers model refresh.
- **Separation of Concerns**: Data API (Railway) handles lottery data, Model API (Railway) handles predictions, Orchestrator coordinates both.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA REFRESH SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                      ┌──────────────────────────────┐ │
│  │   ORCHESTRATOR   │                      │         DATA API             │ │
│  │  orchestrator.py │ ──── GET ──────────▶ │  (Railway - lotto-api)       │ │
│  │                  │   /sorteos/recientes │  Historical lottery draws    │ │
│  │  Runs every      │ ◀─── JSON ────────── │                              │ │
│  │  60 minutes      │                      └──────────────────────────────┘ │
│  │                  │                                                        │
│  │  Checks for      │                      ┌──────────────────────────────┐ │
│  │  new draws       │                      │        MODEL API             │ │
│  │                  │ ──── POST ─────────▶ │  (Railway - web-production)  │ │
│  │  Triggers        │   /admin/retrain     │  LSTM + Statistical Models   │ │
│  │  refresh         │ ◀─── JSON ────────── │                              │ │
│  └──────────────────┘                      └──────────────────────────────┘ │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                        │
│  │ last_processed_  │  Tracks the last draw date processed                  │
│  │ draw.json        │  to avoid duplicate refreshes                         │
│  └──────────────────┘                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Update Frequency

| Setting | Value | Reason |
|---------|-------|--------|
| **Check Interval** | 24 hours | Once daily - sufficient for 3x weekly draws |
| **Lottery Schedule** | Mon, Thu, Sat | Spanish Lotería Primitiva draw days |
| **Data Source** | `/sorteos/recientes` | Returns last 14 draws |

### How It Works

1. **Poll Data API**: Every 24 hours, orchestrator calls the Data API to get recent draws
2. **Compare Dates**: Checks if any draws are newer than `last_processed_draw.json`
3. **Trigger Refresh**: If new draws found, calls `POST /admin/retrain` on Model API
4. **Update Tracker**: Saves the newest draw date to prevent duplicate refreshes

### Running the Orchestrator

```bash
# Run locally
python orchestrator.py

# Run in background (Linux/Mac)
nohup python orchestrator.py &

# Run as a service (production)
# Deploy to Railway/Render as a separate worker process
```

### Admin Endpoint

The Model API exposes an admin endpoint for manual or automated refresh:

```bash
# Trigger manual refresh
curl -X POST https://web-production-09cd3.up.railway.app/admin/retrain

# Response
{
  "status": "ok",
  "message": "Data refreshed and statistical scores recomputed",
  "draws_loaded": 1847,
  "timestamp": "2025-12-11T10:30:00.000000"
}
```

<br>

---

<br>

## � How It Works

### The Prediction Pipeline

<div align="center">

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│ Historical │───▶│   Binary   │───▶│    LSTM    │───▶│  Weighted  │───▶│    Top     │
│    Data    │    │  Encoding  │    │  Ensemble  │    │   Fusion   │    │  Numbers   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘    └────────────┘
                                           │                 ▲
                                           │                 │
                                    ┌──────┴──────┐   ┌──────┴──────┐
                                    │ Statistical │───│   Combine   │
                                    │   Scores    │   │  60% + 40%  │
                                    └─────────────┘   └─────────────┘
```

</div>

<br>

### Step 1: Binary Encoding

Each lottery draw is converted to a 49-dimensional binary vector:

```python
# Draw: [7, 14, 21, 28, 35, 42]
# Binary Vector (size 49):
[0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
#            ↑             ↑             ↑             ↑             ↑             ↑
#         pos 7         pos 14        pos 21        pos 28        pos 35        pos 42
```

### Step 2: Sliding Window

The model analyzes the last 20 draws to predict the next:

```python
WINDOW_LENGTH = 20   # Temporal context window
LOTTERY_SIZE = 49    # Total numbers (1-49)

# Input shape: (batch_size, 20, 49)
recent_draws = binary_dataset[-20:].reshape(1, 20, 49)
```

### Step 3: LSTM Ensemble Prediction

Five independently trained models vote on predictions:

```python
# Ensemble averaging
lstm_predictions = []
for model in ensemble_models:  # 5 models
    pred = model.predict(recent_draws, verbose=0)
    lstm_predictions.append(pred)

ensemble_output = np.mean(lstm_predictions, axis=0)[0]  # Shape: (49,)
```

### Step 4: Statistical Scoring

Two complementary statistical measures:

```python
# Frequency Score: How often each number appeared (last 100 draws)
frequency_score = appearance_count / 100

# Recency Score: How recently each number was drawn
recency_score = 1 / (draws_since_last_appearance + 1)

# Combined Statistical Score
statistical_score = 0.7 * frequency_score + 0.3 * recency_score
```

### Step 5: Weighted Fusion

Final prediction combines both approaches:

```python
# Normalize scores
lstm_normalized = ensemble_output / ensemble_output.sum()
stat_normalized = statistical_score / statistical_score.sum()

# Weighted combination
final_scores = 0.6 * lstm_normalized + 0.4 * stat_normalized

# Get top predictions
top_numbers = np.argsort(final_scores)[-15:][::-1]
```

<br>

---

<br>

## 🧠 Model Deep Dive

### Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL LSTM MODEL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Layer                                                    │
│   └── Shape: (20 timesteps, 49 features)                        │
│                          ↓                                       │
│   Bidirectional LSTM Layer 1                                    │
│   └── Units: 128 (64 forward + 64 backward)                     │
│   └── Return Sequences: True                                    │
│                          ↓                                       │
│   Dropout Layer                                                  │
│   └── Rate: 0.3 (30% neurons disabled)                          │
│                          ↓                                       │
│   Bidirectional LSTM Layer 2                                    │
│   └── Units: 64 (32 forward + 32 backward)                      │
│   └── Return Sequences: False                                   │
│                          ↓                                       │
│   Dropout Layer                                                  │
│   └── Rate: 0.3                                                  │
│                          ↓                                       │
│   Dense Layer                                                    │
│   └── Units: 128                                                 │
│   └── Activation: ReLU                                          │
│                          ↓                                       │
│   Dropout Layer                                                  │
│   └── Rate: 0.2                                                  │
│                          ↓                                       │
│   Output Layer                                                   │
│   └── Units: 49                                                  │
│   └── Activation: Sigmoid (probability per number)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **Bidirectional LSTM** | Process sequences in both directions | Captures forward and backward temporal patterns |
| **Ensemble (5 models)** | Multiple independently trained models | Reduces variance, improves stability |
| **Dropout Layers** | Randomly disable neurons during training | Prevents overfitting to training data |
| **Sigmoid Output** | Probability for each number | Independent probability per number (multi-label) |
| **Statistical Fusion** | Combine with frequency analysis | Balances learned patterns with empirical data |

### Training Configuration

```python
# Model compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Multi-label classification
    metrics=['accuracy']
)

# Training parameters
epochs = 100
batch_size = 32
validation_split = 0.2
```

<br>

---

<br>

## 🛠️ Technology Stack

<div align="center">

| Category | Technology | Version | Purpose |
|:--------:|:----------:|:-------:|:-------:|
| **Language** | Python | 3.11.9 | Core programming language |
| **ML Framework** | TensorFlow | 2.17.0 | Deep learning & model training |
| **Keras** | tf-keras | 2.17.0 | High-level neural network API |
| **Web Framework** | FastAPI | 0.104.1 | REST API development |
| **ASGI Server** | Uvicorn | 0.24.0 | Production-grade async server |
| **Data Processing** | NumPy | 1.26.4 | Numerical computations |
| **Data Analysis** | Pandas | 2.0.3 | Data manipulation |
| **Validation** | Pydantic | 2.5.0 | Request/response validation |
| **Model Storage** | H5py | 3.11.0 | Keras model serialization |
| **Serialization** | Protobuf | 3.20.3 | TensorFlow model format |

</div>

<br>

---

<br>

## 📁 Project Structure

```
lottery-prediction-api/
│
├── 📄 api.py                    # Main FastAPI application (Model API)
├── 📄 orchestrator.py           # Data refresh scheduler (checks every 24 hours)
│
├── 📁 models/                   # Trained ML models
│   ├── lstm_model_1.keras       # Ensemble model 1 (~3.2 MB)
│   ├── lstm_model_2.keras       # Ensemble model 2 (~3.2 MB)
│   ├── lstm_model_3.keras       # Ensemble model 3 (~3.2 MB)
│   ├── lstm_model_4.keras       # Ensemble model 4 (~3.2 MB)
│   ├── lstm_model_5.keras       # Ensemble model 5 (~3.2 MB)
│   ├── lottery_lstm_model.keras # Base model (~3.2 MB)
│   └── statistical_data.pkl     # Historical data + stats (~393 KB)
│
├── 📄 last_processed_draw.json  # Tracks last processed draw date (auto-generated)
├── 📄 requirements.txt          # Python dependencies
├── 📄 runtime.txt               # Python version (Heroku)
├── 📄 .python-version           # Python version (Railway/pyenv)
├── 📄 nixpacks.toml             # Railway build configuration
├── 📄 Procfile                  # Process file (Heroku/Railway)
├── 📄 render.yaml               # Render.com configuration
├── 📄 .gitignore                # Git ignore rules
├── 📄 LICENSE                   # MIT License
└── 📄 README.md                 # Documentation (this file)
```

<br>

---

<br>

## 🚀 Getting Started

### Prerequisites

- **Python 3.11.9** or higher
- **pip** (Python package manager)
- **Git** (version control)
- **8GB RAM** minimum (for TensorFlow)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Najam0786/lottery-prediction-api.git
cd lottery-prediction-api
```

#### 2. Create Virtual Environment

<details>
<summary><b>Windows</b></summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

</details>

<details>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```

</details>

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the Server

```bash
# Option 1: Direct Python
python api.py

# Option 2: Uvicorn with hot reload
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### 5. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# Get predictions
curl "http://localhost:8000/predict?top_n=10&n_combinations=5"
```

<br>

---

<br>

## � Mobile App Integration

Perfect for iOS, Android, and web applications! Our API is designed with mobile-first principles.

### 🚀 Quick Integration

**Base URLs:**
```
Production: https://web-production-09cd3.up.railway.app
Local:      http://localhost:8000
```

### 📋 Available Endpoints

| Endpoint | Method | Use Case | Mobile Friendly |
|----------|--------|----------|-----------------|
| `/user/predict` | POST | Get AI-generated predictions | ✅ JSON body |
| `/user/score-combinations` | POST | Score user combinations | ✅ JSON body |
| `/health` | GET | Check API status | ✅ Lightweight |
| `/predict` | GET | Quick predictions | ⚠️ Query params |

### 💻 Code Examples

#### Swift (iOS)
```swift
struct LotteryService {
    static let baseURL = "https://web-production-09cd3.up.railway.app"
    
    func getPredictions(top_n: Int, combinations: Int) async throws -> PredictionResponse {
        let url = URL(string: "\(baseURL)/user/predict")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["top_n": top_n, "n_combinations": combinations]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(PredictionResponse.self, from: data)
    }
    
    func scoreCombinations(_ combinations: [[Int]]) async throws -> ScoreResponse {
        let url = URL(string: "\(baseURL)/user/score-combinations")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body = ["combinations": combinations]
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(ScoreResponse.self, from: data)
    }
}
```

#### Kotlin (Android)
```kotlin
class LotteryApiService {
    private val client = OkHttpClient()
    private val gson = Gson()
    private val baseURL = "https://web-production-09cd3.up.railway.app"
    
    suspend fun getPredictions(topN: Int, nCombinations: Int): PredictionResponse {
        val url = "$baseURL/user/predict"
        val body = JSONObject().apply {
            put("top_n", topN)
            put("n_combinations", nCombinations)
        }
        
        val request = Request.Builder()
            .url(url)
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
            
        val response = client.newCall(request).execute()
        return gson.fromJson(response.body?.string(), PredictionResponse::class.java)
    }
    
    suspend fun scoreCombinations(combinations: List<List<Int>>): ScoreResponse {
        val url = "$baseURL/user/score-combinations"
        val body = JSONObject().apply {
            put("combinations", JSONArray(combinations))
        }
        
        val request = Request.Builder()
            .url(url)
            .post(body.toString().toRequestBody("application/json".toMediaType()))
            .build()
            
        val response = client.newCall(request).execute()
        return gson.fromJson(response.body?.string(), ScoreResponse::class.java)
    }
}
```

#### JavaScript (Web/React)
```javascript
class LotteryAPI {
    constructor(baseURL = 'https://web-production-09cd3.up.railway.app') {
        this.baseURL = baseURL;
    }
    
    async getPredictions(topN = 15, nCombinations = 10) {
        const response = await fetch(`${this.baseURL}/user/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ top_n: topN, n_combinations: nCombinations })
        });
        return await response.json();
    }
    
    async scoreCombinations(combinations) {
        const response = await fetch(`${this.baseURL}/user/score-combinations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ combinations })
        });
        return await response.json();
    }
    
    async checkHealth() {
        const response = await fetch(`${this.baseURL}/health`);
        return await response.json();
    }
}

// Usage
const lottery = new LotteryAPI();
const predictions = await lottery.getPredictions(10, 5);
const scores = await lottery.scoreCombinations([[7,14,21,28,35,42]]);
```

### 🔧 Best Practices

✅ **Use POST endpoints** for mobile apps (`/user/predict`, `/user/score-combinations`)
✅ **Implement retry logic** for network failures
✅ **Cache predictions** for offline use
✅ **Handle rate limits** gracefully
✅ **Validate user input** before sending to API

### 🚨 Error Handling

```swift
// Swift Example
do {
    let predictions = try await lotteryService.getPredictions(top_n: 10, combinations: 5)
    // Update UI with predictions
} catch {
    if let urlError = error as? URLError {
        switch urlError.code {
        case .notConnectedToInternet:
            showNetworkError()
        case .timedOut:
            showTimeoutError()
        default:
            showGenericError()
        }
    }
}
```

### 📊 Response Models

```swift
struct PredictionResponse: Codable {
    let top_numbers: [NumberPrediction]
    let combinations: [[Int]]
    let metadata: Metadata
}

struct ScoreResponse: Codable {
    let scored_combinations: [CombinationScore]
    let metadata: Metadata
}
```

---

<br>

## �📡 API Reference

### Base URL

```
Production: https://web-production-09cd3.up.railway.app
Local:      http://localhost:8000
```

<br>

### `GET /`

Returns API information and status.

<details>
<summary><b>Response Example</b></summary>

```json
{
  "name": "Lotería Primitiva Prediction API",
  "version": "1.0.0",
  "status": "active",
  "models": "loaded",
  "data": "loaded",
  "documentation": {
    "swagger_ui": "/docs",
    "redoc": "/redoc",
    "openapi_json": "/openapi.json"
  },
  "endpoints": {
    "/predict": "Get lottery number predictions",
    "/user/predict": "User-facing predictions (POST) - for mobile/web apps",
    "/user/score-combinations": "Score user combinations with explanations (POST) - NEW!",
    "/health": "Health check",
    "/admin/retrain": "Trigger data refresh (POST)",
    "/docs": "Interactive API documentation (Swagger UI)",
    "/redoc": "Alternative API documentation"
  }
}
```

</details>

<br>

### `GET /health`

Health check endpoint for monitoring and load balancers.

<details>
<summary><b>Response Example</b></summary>

```json
{
  "status": "healthy",
  "models_loaded": 5,
  "data_loaded": true,
  "model_type": "Bidirectional LSTM + Statistical Ensemble",
  "ensemble_models": 5,
  "accuracy": "87.76%",
  "timestamp": "2024-12-02T12:00:00.000000"
}
```

</details>

<br>

### `GET /predict`

Get lottery number predictions.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | integer | 15 | Number of top predictions to return (1-49) |
| `n_combinations` | integer | 10 | Number of 6-number combinations to generate |

#### Example Request

```bash
curl "https://web-production-09cd3.up.railway.app/predict?top_n=10&n_combinations=5"
```

<details>
<summary><b>Response Example</b></summary>

```json
{
  "top_numbers": [
    {
      "number": 23,
      "score": 0.0847,
      "lstm_score": 0.0312,
      "stat_score": 0.1523
    },
    {
      "number": 7,
      "score": 0.0789,
      "lstm_score": 0.0298,
      "stat_score": 0.1412
    }
  ],
  "combinations": [
    [7, 14, 23, 31, 38, 45],
    [3, 12, 23, 28, 35, 47],
    [7, 19, 25, 33, 41, 49],
    [5, 14, 22, 30, 38, 44],
    [8, 16, 23, 29, 37, 46]
  ],
  "metadata": {
    "model_type": "Bidirectional LSTM + Statistical Ensemble",
    "ensemble_models": 5,
    "accuracy": "87.76%",
    "timestamp": "2024-12-02T12:00:00.000000",
    "parameters": {
      "top_n": 10,
      "n_combinations": 5
    }
  }
}
```

</details>

<br>

### `POST /user/predict`

User-facing prediction endpoint for mobile and web applications. Accepts JSON body instead of query parameters.

#### Request Body

```json
{
  "top_n": 10,
  "n_combinations": 5
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | integer | 15 | Number of top predictions (1-49) |
| `n_combinations` | integer | 10 | Number of combinations (1-100) |

#### Example Request

```bash
curl -X POST "https://web-production-09cd3.up.railway.app/user/predict" \
  -H "Content-Type: application/json" \
  -d '{"top_n": 10, "n_combinations": 5}'
```

<details>
<summary><b>Response Example</b></summary>

```json
{
  "top_numbers": [
    {"number": 23, "score": 0.0847, "lstm_score": 0.0312, "stat_score": 0.1523},
    {"number": 7, "score": 0.0789, "lstm_score": 0.0298, "stat_score": 0.1412}
  ],
  "combinations": [
    [7, 14, 23, 31, 38, 45],
    [3, 12, 23, 28, 35, 47]
  ],
  "metadata": {
    "model_type": "Bidirectional LSTM + Statistical Ensemble",
    "ensemble_models": 5,
    "accuracy": "87.76%",
    "timestamp": "2025-12-11T12:00:00.000000",
    "parameters": {"top_n": 10, "n_combinations": 5}
  }
}
```

</details>

<br>

### `POST /user/score-combinations` ⭐ **NEW!**

Score user-provided lottery combinations with detailed rational explanations. Perfect for validating your own number choices!

#### Request Body

```json
{
  "combinations": [
    [7, 14, 21, 28, 35, 42],
    [1, 2, 3, 4, 5, 6]
  ]
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `combinations` | array | List of lottery combinations (each with 6 unique numbers 1-49) |

#### Example Request

```bash
curl -X POST "https://web-production-09cd3.up.railway.app/user/score-combinations" \
  -H "Content-Type: application/json" \
  -d '{"combinations": [[7,14,21,28,35,42], [1,2,3,4,5,6]]}'
```

<details>
<summary><b>Response Example</b></summary>

```json
{
  "scored_combinations": [
    {
      "combination": [7, 14, 21, 28, 35, 42],
      "score": 78.5,
      "individual_scores": [82.1, 75.3, 79.8, 76.2, 81.4, 73.9],
      "rational": "Strong numbers: 7, 21, 35 (high LSTM/statistical confidence) | Good number distribution across low/mid/high ranges | Excellent combination with strong statistical backing"
    },
    {
      "combination": [1, 2, 3, 4, 5, 6],
      "score": 25.3,
      "individual_scores": [45.2, 38.1, 42.7, 35.9, 48.3, 52.1],
      "rational": "Weak numbers: 2, 4 (low historical frequency) | Poor number distribution - concentrated in low range | Contains 5 consecutive pairs (statistically rare) | Shows arithmetic pattern (reduces randomness) | Weak combination with multiple statistical issues"
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
```

</details>

#### Scoring Features

🧠 **Individual Number Analysis**
- LSTM confidence scores for each number
- Statistical frequency and recency analysis
- Combined weighted scoring (60% LSTM + 40% Statistical)

📊 **Pattern Detection**
- Consecutive number analysis
- Arithmetic pattern detection
- Number distribution across ranges

💡 **Rational Explanations**
- Detailed breakdown of why each combination scored as it did
- Identifies strong and weak numbers
- Provides actionable insights for number selection

#### Score Interpretation

| Score Range | Assessment | Description |
|-------------|------------|-------------|
| 70-100 | Excellent | Strong statistical backing, recommended |
| 50-69 | Good | Moderate statistical support, viable option |
| 30-49 | Fair | Some statistical weaknesses, use with caution |
| 0-29 | Weak | Multiple statistical issues, not recommended |

<br>

### `POST /admin/retrain`

Trigger data refresh and recompute statistical scores. Used by the orchestrator for automated updates.

#### Example Request

```bash
curl -X POST "https://web-production-09cd3.up.railway.app/admin/retrain"
```

<details>
<summary><b>Response Example</b></summary>

```json
{
  "status": "ok",
  "message": "Data refreshed and statistical scores recomputed",
  "draws_loaded": 1847,
  "timestamp": "2025-12-11T12:00:00.000000"
}
```

</details>

<br>

---

<br>

## ☁️ Cloud Deployment

### 🚂 Railway (Recommended)

Railway offers the simplest deployment experience with automatic builds.

#### Configuration Files Required

**`.python-version`**
```
3.11.9
```

**`nixpacks.toml`**
```toml
[phases.setup]
nixPkgs = ["python311"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[start]
cmd = "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"
```

#### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**
   - Go to [railway.app](https://railway.app)
   - Click **"New Project"** → **"Deploy from GitHub repo"**
   - Select your repository
   - Railway auto-detects Python and deploys

3. **Verify Deployment**
   ```bash
   curl https://your-app.up.railway.app/health
   ```

<br>

### 🎨 Render

Render provides free tier hosting with automatic SSL.

#### Configuration File

**`render.yaml`**
```yaml
services:
  - type: web
    name: lottery-prediction-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
    python:
      version: 3.11.9
```

#### Deployment Steps

1. Go to [render.com](https://render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Render detects `render.yaml` and configures automatically
5. Click **"Create Web Service"**

<br>

### 🟣 Heroku

Heroku uses Procfile for process management.

#### Configuration Files

**`Procfile`**
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

**`runtime.txt`**
```
python-3.11.9
```

#### Deployment Steps

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login to Heroku
heroku login

# Create app
heroku create lottery-prediction-api

# Deploy
git push heroku main

# Open app
heroku open
```

<br>

### 🌐 Google Cloud Run

Deploy as a containerized service on Google Cloud.

#### Create Dockerfile

```dockerfile
FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Deployment Steps

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud run deploy lottery-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

<br>

### ⚡ AWS Lambda (with Mangum)

Deploy as a serverless function on AWS.

#### Install Mangum Adapter

```bash
pip install mangum
```

#### Modify `api.py`

```python
# Add at the end of api.py
from mangum import Mangum
handler = Mangum(app)
```

#### Deploy with AWS SAM

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  LotteryAPI:
    Type: AWS::Serverless::Function
    Properties:
      Handler: api.handler
      Runtime: python3.11
      Timeout: 300
      MemorySize: 1024
      Events:
        Api:
          Type: HttpApi
```

```bash
sam build
sam deploy --guided
```

<br>

### 🔷 Azure App Service

Deploy to Microsoft Azure.

#### Create `startup.txt`

```
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### Deployment Steps

```bash
# Install Azure CLI
# https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

# Login
az login

# Create resource group
az group create --name lottery-rg --location eastus

# Create App Service plan
az appservice plan create --name lottery-plan --resource-group lottery-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group lottery-rg --plan lottery-plan --name lottery-prediction-api --runtime "PYTHON:3.11"

# Deploy
az webapp up --name lottery-prediction-api --resource-group lottery-rg
```

<br>

---

<br>

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `TF_ENABLE_ONEDNN_OPTS` | 1 | TensorFlow optimizations |

### File Reference

| File | Platform | Purpose |
|------|----------|---------|
| `requirements.txt` | All | Python dependencies |
| `.python-version` | Railway, pyenv | Python version |
| `runtime.txt` | Heroku | Python version |
| `nixpacks.toml` | Railway | Build configuration |
| `Procfile` | Heroku, Railway | Process command |
| `render.yaml` | Render | Service configuration |

<br>

---

<br>

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>❌ TensorFlow Import Error</b></summary>

```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution:**
```bash
pip install tensorflow==2.17.0 tf-keras==2.17.0
```

</details>

<details>
<summary><b>❌ NumPy Compatibility Error</b></summary>

```
AttributeError: module 'numpy' has no attribute '_core'
```

**Solution:** The API includes a custom `RenameUnpickler` class that handles this automatically. Ensure you're using NumPy 1.26.4.

</details>

<details>
<summary><b>❌ Railway Using Python 3.14</b></summary>

**Solution:** Create all version specification files:
```bash
echo "3.11.9" > .python-version
echo "3.11.9" > runtime.txt
```

And ensure `nixpacks.toml` has:
```toml
[phases.setup]
nixPkgs = ["python311"]
```

</details>

<details>
<summary><b>❌ Models Not Loading</b></summary>

```
FileNotFoundError: models/lstm_model_1.keras
```

**Solution:** Ensure models are tracked in Git:
```bash
git add models/
git commit -m "Add trained models"
git push
```

</details>

<details>
<summary><b>❌ Memory Error on Deployment</b></summary>

**Solution:** Increase instance memory:
- Railway: Upgrade to Pro plan
- Render: Use Standard instance
- Heroku: Use Standard-2X dyno

</details>

<br>

---

<br>

## ⚡ Performance Optimization

### Production Recommendations

1. **Enable Model Caching**
   ```python
   # Models are loaded once at startup via lifespan
   @asynccontextmanager
   async def lifespan(app: FastAPI):
       load_models()  # Load once
       yield
   ```

2. **Use Gunicorn with Uvicorn Workers**
   ```bash
   gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Enable Response Compression**
   ```python
   from fastapi.middleware.gzip import GZipMiddleware
   app.add_middleware(GZipMiddleware, minimum_size=1000)
   ```

4. **Add Redis Caching** (for high traffic)
   ```python
   import redis
   cache = redis.Redis(host='localhost', port=6379)
   ```

<br>

---

<br>

## 🤝 Contributing

Contributions make the open-source community amazing! Any contributions are **greatly appreciated**.

### How to Contribute

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Contribution Ideas

- [ ] Add Transformer-based model architecture
- [ ] Implement real-time data fetching from lottery APIs
- [ ] Create React/Vue frontend dashboard
- [ ] Add support for other lottery types (EuroMillions, Powerball)
- [ ] Implement A/B testing for model comparison
- [ ] Add Prometheus metrics endpoint
- [ ] Create Docker Compose setup

<br>

---

<br>

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

```
MIT License

Copyright (c) 2024 Najam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<br>

---

<br>

## ⚠️ Disclaimer

<div align="center">

**This project is for educational and entertainment purposes only.**

Lottery outcomes are random events governed by probability.
No prediction system can guarantee winning numbers.
Please gamble responsibly and within your means.

</div>

<br>

---

<br>

## 🙏 Acknowledgments

<div align="center">

| | |
|:---:|:---:|
| [TensorFlow](https://tensorflow.org) | Deep learning framework |
| [FastAPI](https://fastapi.tiangolo.com) | Modern Python web framework |
| [Railway](https://railway.app) | Cloud deployment platform |
| [Render](https://render.com) | Cloud hosting service |
| [NumPy](https://numpy.org) | Numerical computing |
| [Pydantic](https://pydantic.dev) | Data validation |

</div>

<br>

---

<br>

<div align="center">

### 👨‍💻 Author

**Najam**

[![GitHub](https://img.shields.io/badge/GitHub-Najam0786-181717?style=for-the-badge&logo=github)](https://github.com/Najam0786)

<br>

---

<br>

<sub>If you found this project helpful, please consider giving it a ⭐</sub>

<br>

**[⬆ Back to Top](#-lottery-prediction-api)**

</div>

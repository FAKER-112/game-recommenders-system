# Game Recommendation System 🎮

A comprehensive recommendation system for Steam games built with TensorFlow, featuring three distinct machine learning models (Autoencoder, Matrix Factorization, and TFRS) and a production-ready Flask API.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Overview

This project implements a multi-model game recommendation system that analyzes user-game interactions from Steam to provide:
- **Personalized Recommendations**: Suggest games based on user play history
- **Content-Based Similarity**: Find similar games based on genres and tags
- **Hybrid Approaches**: Combine collaborative and content-based filtering

The system supports three different recommendation models:
1. **Autoencoder** - Content-based recommendations using game features
2. **Matrix Factorization** - Collaborative filtering based on user-item interactions
3. **TFRS (TensorFlow Recommenders)** - Scalable two-tower retrieval model

## Features

✨ **Multiple Recommendation Strategies**
- User-based collaborative filtering (MF, TFRS)
- Item-based content similarity (Autoencoder)
- Hybrid recommendation approaches

📊 **Complete ML Pipeline**
- Automated data ingestion from public datasets
- Feature engineering and preprocessing
- Model training with MLflow experiment tracking
- Comprehensive evaluation metrics

🚀 **Production-Ready API**
- RESTful Flask API with CORS support
- Batch recommendation endpoints
- Pagination and search functionality
- Health monitoring and error handling

📈 **Experiment Tracking**
- MLflow integration for all training runs
- Hyperparameter logging
- Metric visualization and comparison

🐳 **Containerization**
- Docker support for easy deployment
- Environment consistency

## Project Structure

```
game-recommenders-system/
├── app.py                          # Flask API server
├── requirements.txt                # Python dependencies
├── configs/                        # Configuration files
│   ├── config.yaml                 # Data pipeline config
│   ├── model_params.yaml           # Model hyperparameters
│   └── pipeline_params.yaml        # Pipeline orchestration
├── src/
│   ├── data/                       # Data processing modules
│   │   ├── load_data.py            # Data ingestion
│   │   ├── clean_data.py           # Data cleaning
│   │   └── feature_engineering.py  # Feature creation
│   ├── models/                     # Model definitions
│   │   ├── build_model.py          # Model architectures
│   │   ├── train_model.py          # Training logic
│   │   └── evaluate_model.py       # Evaluation metrics
│   ├── pipeline/                   # Pipeline orchestration
│   │   ├── train_pipeline.py       # End-to-end training
│   │   ├── predict_pipeline.py     # Inference engine
│   │   └── evaluate_pipeline.py    # Model evaluation
│   └── utils/                      # Utility functions
│       ├── logger.py               # Logging configuration
│       └── exception.py            # Custom exceptions
├── data/                           # Data directories
│   ├── raw/                        # Raw downloaded data
│   └── processed/                  # Processed datasets
├── artifacts/                      # Model artifacts
│   ├── models/                     # Trained models
│   └── context/                    # Encoders and vocabularies
├── mlruns/                         # MLflow experiment logs
├── notebooks/                      # Jupyter notebooks
├── templates/                      # HTML templates
└── static/                         # Static files (CSS, JS)
```

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Python package manager
- **Git**: Version control
- **Disk Space**: ~5GB for datasets and models
- **RAM**: Minimum 8GB (16GB recommended for training)
- **GPU**: Optional but recommended for faster training

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project_007
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tensorflow_recommenders as tfrs; print('TFRS installed successfully')"
```

## Quick Start

### Option 1: Use Pre-trained Models (Fastest)

If you have pre-trained models in the `artifacts/` directory:

```bash
# Start the Flask API server
python app.py
```

The API will be available at `http://localhost:5000`

### Option 2: Train from Scratch

#### Step 1: Run Complete Training Pipeline

```bash
# Train all models (this will take 30-60 minutes)
python src/pipeline/train_pipeline.py
```

This will:
1. Download raw Steam game data (~2GB)
2. Clean and process the data
3. Engineer features
4. Train all three models
5. Save models and artifacts
6. Log experiments to MLflow

#### Step 2: Evaluate Models

```bash
# Evaluate trained models
python src/pipeline/evaluate_pipeline.py
```

#### Step 3: Start the API Server

```bash
# Launch Flask application
python app.py
```

### Option 3: Step-by-Step Training

For more control, run each stage separately:

```bash
# 1. Data Ingestion
python -c "from src.data.load_data import LoadDataService; LoadDataService().run()"

# 2. Data Cleaning
python -c "from src.data.clean_data import CleanDataService; CleanDataService().run()"

# 3. Feature Engineering
python -c "from src.data.feature_engineering import FeatureEngineeringService; FeatureEngineeringService().run()"

# 4. Train specific model
python -c "from src.models.train_model import ModelTrainingService; ModelTrainingService(model_type='tfrs').run()"

# 5. Evaluate model
python -c "from src.models.evaluate_model import ModelEvaluationService; ModelEvaluationService(model_type='tfrs').run()"
```

## Usage Guide

### Training Models

#### Using Python API

```python
from src.pipeline.train_pipeline import TrainingPipeline

# Train Matrix Factorization model
mf_pipeline = TrainingPipeline(model_type='matrix_factorization')
mf_pipeline.train_model()

# Train TFRS model
tfrs_pipeline = TrainingPipeline(model_type='tfrs')
tfrs_pipeline.train_model()

# Train Autoencoder model
ae_pipeline = TrainingPipeline(model_type='autoencoder')
ae_pipeline.train_model()
```

### Making Predictions

#### Using Python API

```python
from src.pipeline.predict_pipeline import PredictionPipeline

# Initialize predictor
predictor = PredictionPipeline(model_type='tfrs')

# Get user recommendations
recommendations = predictor.recommend(
    user_id='76561197970982479',
    n_rec=10
)
print("Recommended games:", recommendations)

# Find similar items
similar_games = predictor.get_similar_items(
    item_name='Counter-Strike',
    k=5
)
print("Similar games:", similar_games)
```

#### Using REST API

```bash
# Get user recommendations
curl -X POST http://localhost:5000/recommend_user \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "tfrs",
    "user_id": "76561197970982479",
    "n_rec": 10
  }'

# Find similar items
curl -X POST http://localhost:5000/recommend_item \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "autoencoder",
    "item_name": "Counter-Strike",
    "k": 5
  }'
```

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
# View experiments, compare metrics, and analyze runs
```

## API Documentation

### Base URL

```
http://localhost:5000
```

### Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": ["tfrs"],
  "data_loaded": {"games": true, "users": true}
}
```

#### 2. Get User List

```http
GET /api/userlist
```

**Response:**
```json
{
  "status": "success",
  "users": ["user1", "user2", ...],
  "count": 100
}
```

#### 3. Get Game Data

```http
GET /api/gamedata?start=0&end=10&search=counter
```

**Parameters:**
- `start` (optional): Starting index (default: 0)
- `end` (optional): Ending index (default: 10)
- `search` (optional): Search term for filtering

**Response:**
```json
{
  "status": "success",
  "games": [{...}],
  "total_games": 1000,
  "has_more": true
}
```

#### 4. User Recommendations

```http
POST /recommend_user
Content-Type: application/json

{
  "model_name": "tfrs",
  "user_id": "76561197970982479",
  "n_rec": 10
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": "76561197970982479",
  "model_used": "tfrs",
  "recommendations": ["Game 1", "Game 2", ...],
  "recommendations_with_details": [{...}],
  "count": 10
}
```

#### 5. Item Similarity

```http
POST /recommend_item
Content-Type: application/json

{
  "model_name": "autoencoder",
  "item_name": "Counter-Strike",
  "k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "query_item": "Counter-Strike",
  "similar_items": ["Game 1", "Game 2", ...],
  "similar_items_with_details": [{...}],
  "count": 5
}
```

#### 6. Batch Recommendations

```http
POST /batch_recommend
Content-Type: application/json

{
  "model_name": "tfrs",
  "user_ids": ["user1", "user2", "user3"],
  "n_rec": 10
}
```

**Response:**
```json
{
  "status": "success",
  "results": {
    "user1": ["Game 1", "Game 2", ...],
    "user2": ["Game 3", "Game 4", ...]
  },
  "total_users": 2,
  "successful": 2
}
```

## Configuration

### Model Hyperparameters

Edit `configs/model_params.yaml` to adjust model settings:

```yaml
model_params:
  autoencoder:
    encoding_dim: 64
    hidden_layers: [512, 256]
  
  matrix_factorization:
    embedding_size: 50
    hidden_layers: [512, 256]
  
  tfrs:
    embedding_size: 50
    hidden_layers: [512, 256]
```

### Data Pipeline Settings

Edit `configs/config.yaml` for data processing:

```yaml
data_ingestion:
  user_item_dataset_download_url: "<URL>"
  raw_data_dir: "data/raw"

data_cleaning:
  processed_dir: "data/processed"
```

### Training Configuration

Edit `configs/model_params.yaml` for training settings:

```yaml
model_training:
  epochs: 10
  batch_size: 32
  root_dir: 'artifacts/models'
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_models.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Development Server

```bash
# Run Flask in debug mode
export FLASK_ENV=development  # Linux/Mac
set FLASK_ENV=development     # Windows

python app.py
```

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t game-recommender:latest .

# Run container
docker run -p 5000:5000 game-recommender:latest
```

### Production Considerations

1. **Environment Variables**:
   ```bash
   export FLASK_ENV=production
   export MLFLOW_TRACKING_URI=<your-mlflow-server>
   ```

2. **WSGI Server**:
   ```bash
   # Use Gunicorn for production
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Model Serving**:
   - Consider TensorFlow Serving for high-throughput
   - Use Redis for caching recommendations
   - Implement request queuing for batch processing

4. **Monitoring**:
   - Set up application logging
   - Monitor API response times
   - Track model performance metrics

## Model Comparison

| Model | Use Case | Latency | Best For |
|-------|----------|---------|----------|
| **TFRS** | User recommendations | ~1-5ms | Large-scale retrieval, known users |
| **Matrix Factorization** | User recommendations | ~5-20ms | Personalized ranking |
| **Autoencoder** | Item similarity | ~10-50ms | Content-based, cold-start items |

## Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Ensure models are trained
python src/pipeline/train_pipeline.py
```

**2. Data not found**
```bash
# Re-run data ingestion
python -c "from src.data.load_data import LoadDataService; LoadDataService().run()"
```

**3. Port already in use**
```bash
# Change port in app.py or kill existing process
lsof -i :5000  # Linux/Mac
netstat -ano | findstr :5000  # Windows
```

**4. Out of memory during training**
```bash
# Reduce batch size in configs/model_params.yaml
# Use smaller hidden layers
# Process data in chunks
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Dataset**: Steam game data from [UCSD Public Datasets](https://cseweb.ucsd.edu/~jmcauley/)
- **Framework**: TensorFlow and TensorFlow Recommenders
- **Experiment Tracking**: MLflow

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using TensorFlow, Flask, and MLflow**
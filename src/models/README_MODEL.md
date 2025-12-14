# Model Training & Evaluation

This directory contains the complete modeling pipeline for the game recommendation system. It supports three different recommendation approaches, each with its own architecture, training strategy, and evaluation metrics.

## Overview

The modeling pipeline consists of three main components:
1. **Model Building** - Defines architectures and prepares data for each model type
2. **Model Training** - Orchestrates training loops, MLflow logging, and model persistence
3. **Model Evaluation** - Computes ranking metrics and validates model performance

## Model Architectures

The system supports three distinct recommendation models:

### 1. Autoencoder (Content-Based)
- **Type**: Neural autoencoder for content similarity
- **Approach**: Reconstructs item feature vectors to learn compressed representations
- **Use Case**: Find similar games based on genres and tags
- **Input**: TF-IDF vectors of item text (genres + tags)
- **Output**: Dense embeddings for similarity computation

### 2. Matrix Factorization (Collaborative Filtering)
- **Type**: Neural collaborative filtering
- **Approach**: Learns latent embeddings for users and items
- **Use Case**: Personalized recommendations based on user-item interactions
- **Input**: User IDs, Item IDs, and implicit ratings (log-transformed playtime)
- **Output**: Rating predictions for user-item pairs

### 3. TFRS Two-Tower (Retrieval Model)
- **Type**: TensorFlow Recommenders retrieval model
- **Approach**: Separate user and item towers for efficient candidate retrieval
- **Use Case**: Scalable retrieval from large item catalogs
- **Input**: User IDs, Item IDs, and item text features
- **Output**: User and item embeddings for similarity-based retrieval

## Modules

### 1. build_model.py

**Purpose**: Factory class for model construction and data preparation.

**Class**: `ModelBuilder`

**Functionality**:

The `ModelBuilder` class follows a factory pattern with paired methods for each model:
- `prepare_data_<model_type>()` - Transforms DataFrames into model-specific formats
- `build_<model_type>_model()` - Defines the neural network architecture

**Data Preparation Methods**:
- `prepare_data_autoencoder()`: Creates TF-IDF vectors from item text
- `prepare_data_mf()`: Encodes user/item IDs and prepares rating arrays
- `prepare_data_tfrs()`: Builds TensorFlow Datasets with vocabularies

**Model Building Methods**:
- `build_autoencoder_model()`: Encoder-decoder architecture with customizable hidden layers
- `build_mf_model()`: User/item embedding layers followed by dense layers
- `build_tfrs_model()`: Two-tower architecture with query and candidate towers

**Configuration**: Uses parameters from `configs/model_params.yaml`

---

### 2. train_model.py

**Purpose**: Model training orchestration with MLflow tracking.

**Class**: `ModelTrainingService`

**Functionality**:
1. **Configuration Loading**: Reads hyperparameters from `configs/model_params.yaml`
2. **Data Loading**: Loads train/test datasets from `data/processed/`
3. **Model Construction**: Uses `ModelBuilder` to create model architecture
4. **Data Preparation**: Applies model-specific preprocessing
5. **Training Loop**: Fits model with validation monitoring
6. **MLflow Logging**: 
   - Logs hyperparameters (epochs, batch size, embedding dimensions)
   - Logs training metrics (loss, accuracy per epoch)
   - Tags experiments with model type
7. **Model Persistence**:
   - Saves models to `artifacts/models/` (H5 or SavedModel format)
   - Saves context artifacts to `artifacts/context/` (encoders, vocabularies, TF-IDF vectorizers)

**Training Strategy**:
- **Autoencoder**: Self-supervised reconstruction of item features
- **Matrix Factorization**: Supervised learning on user-item ratings
- **TFRS**: Retrieval task with factorized top-K metrics

**Output Artifacts**:
- Trained model files (`.h5` or SavedModel directories)
- Feature encoders and vocabularies (`.pkl` files)
- TF-IDF vectorizers for content-based models

---

### 3. evaluate_model.py

**Purpose**: Comprehensive model evaluation with ranking metrics.

**Class**: `ModelEvaluationService`

**Functionality**:

**Ranking Metrics** (for MF & TFRS):
- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Mean Average Precision (MAP@K)**: Average precision across all users
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranked lists

**Content Metrics** (for Autoencoder):
- **Categorized Accuracy**: Percentage of nearest neighbors sharing genres/tags
- **Cosine Similarity**: Average similarity between query and retrieved items

**Evaluation Process**:
1. **Data Loading**: Loads test set and trained model
2. **Prediction Generation**:
   - For MF/TFRS: Generates top-K recommendations for sample users
   - For Autoencoder: Finds K-nearest neighbors using learned embeddings
3. **Metric Calculation**: Computes all relevant metrics
4. **MLflow Logging**: Logs evaluation metrics to experiment tracking
5. **Reporting**: Prints summary statistics and logs to MLflow

**Evaluation Strategy**:
- **Ranking Models**: Test users against candidate items (excluding training items)
- **Content Model**: Test items against item catalog for similarity matching

## Configuration

### Required Config Files

#### 1. configs/model_params.yaml

Contains all model hyperparameters and training settings:

```yaml
model_training:
  epochs: 10                    # Training epochs
  batch_size: 32                # Batch size for training
  root_dir: 'artifacts/models'  # Model save directory
  context_dir: 'artifacts/context'  # Context artifacts directory
  transformed_train_path: "data/processed/train.csv"
  transformed_test_path: "data/processed/test.csv"
  transformed_data_path: "data/processed/transformed_df.csv"

model_params:
  autoencoder:
    input_dim: 5000             # TF-IDF vocabulary size
    encoding_dim: 64            # Embedding dimension
    hidden_layers: [512, 256]   # Hidden layer sizes
    activation: 'relu'
    optimizer: 'adam'
    loss: 'binary_crossentropy'
    
  matrix_factorization:
    num_users: 1000             # Max number of users
    num_items: 500              # Max number of items
    embedding_size: 50          # Embedding dimension
    hidden_layers: [512, 256]
    optimizer: 'adam'
    loss: 'mse'
    
  tfrs:
    num_users: 1000
    num_items: 500
    embedding_size: 50
    hidden_layers: [512, 256]
    optimizer: 'adam'
```

#### 2. configs/config.yaml

Referenced for data paths (used indirectly via training service).

### Environment Variables

- **MLflow Tracking**: Set `MLFLOW_TRACKING_URI` to configure experiment tracking server
- Default: Uses local `mlruns/` directory

## Usage

### Training Models

Run the training pipeline for each model type:

```python
from train_model import ModelTrainingService

# Train Autoencoder
ae_trainer = ModelTrainingService(model_type='autoencoder')
ae_trainer.run()

# Train Matrix Factorization
mf_trainer = ModelTrainingService(model_type='matrix_factorization')
mf_trainer.run()

# Train TFRS
tfrs_trainer = ModelTrainingService(model_type='tfrs')
tfrs_trainer.run()
```

### Evaluating Models

Evaluate trained models on test data:

```python
from evaluate_model import ModelEvaluationService

# Evaluate Autoencoder
ae_eval = ModelEvaluationService(model_type='autoencoder')
ae_eval.run()

# Evaluate Matrix Factorization
mf_eval = ModelEvaluationService(model_type='matrix_factorization')
mf_eval.run()

# Evaluate TFRS
tfrs_eval = ModelEvaluationService(model_type='tfrs')
tfrs_eval.run()
```

### Building Custom Models

Use `ModelBuilder` directly for experimentation:

```python
from build_model import ModelBuilder
import pandas as pd

# Initialize builder
builder = ModelBuilder(config_path='configs/model_params.yaml')

# Load data
df = pd.read_csv('data/processed/train.csv')

# Prepare data for specific model
X, y, context = builder.prepare_data_mf(df)

# Build model
model = builder.build_mf_model()

# Train
model.fit(X, y, epochs=10, batch_size=32)
```

## MLflow Experiment Tracking

All training runs are automatically logged to MLflow with:

**Parameters Logged**:
- Model type
- Epochs
- Batch size
- Embedding dimensions
- Hidden layer configurations
- Optimizer and loss function

**Metrics Logged**:
- Training loss (per epoch)
- Validation loss (per epoch)
- Evaluation metrics (Precision@K, MAP@K, NDCG@K, etc.)

**Artifacts Logged**:
- Trained model files
- Feature encoders and vocabularies
- Configuration files

### Viewing Experiments

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

## Output Structure

```
artifacts/
├── models/
│   ├── autoencoder_model.h5
│   ├── mf_model.h5
│   └── tfrs_model/              # SavedModel directory
│       ├── saved_model.pb
│       └── variables/
└── context/
    ├── user_encoder.pkl
    ├── item_encoder.pkl
    ├── tfidf_vectorizer.pkl
    ├── user_vocab.pkl
    └── item_vocab.pkl
```

## Dependencies

- **TensorFlow** >= 2.x
- **TensorFlow Recommenders** (for TFRS models)
- **Scikit-learn** (for TF-IDF and encoders)
- **MLflow** (for experiment tracking)
- **Pandas** (for data handling)
- **NumPy** (for numerical operations)

## Evaluation Metrics Summary

| Model | Primary Metrics | Use Case |
|-------|----------------|----------|
| **Autoencoder** | Categorized Accuracy, Cosine Similarity | Content similarity, cold-start items |
| **Matrix Factorization** | Precision@K, MAP@K, NDCG@K | Personalized ranking, known users |
| **TFRS** | Factorized Top-K, Precision@K, MAP@K | Large-scale retrieval, hybrid filtering |

## Best Practices

1. **Hyperparameter Tuning**: Modify `configs/model_params.yaml` to experiment with different configurations
2. **MLflow Tracking**: Use unique experiment names for different tuning runs
3. **Data Consistency**: Ensure data pipeline completes before training
4. **Model Selection**: Compare MLflow metrics across models to select best performer
5. **Context Artifacts**: Always save encoders and vocabularies for inference

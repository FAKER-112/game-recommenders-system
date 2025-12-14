Model Builder Script
--------------------
This script defines the `ModelBuilder` class, which acts as a factory and configuration hub
for constructing three distinct types of recommendation system models.

The class handles both the **data preparation** (preprocessing specific to each model architecture)
and the **model architecture definition** (building the TensorFlow/Keras models).

Logic of Operation:
This script relies on the strategy of defining pairs of methods for each model type:
1.  `prepare_data_<model_type>`: Transforms the generic pandas DataFrame input into model-specific
    formats (e.g., TF-IDF matrices, Integer-encoded arrays, or TensorFlow Datasets).
2.  `build_<model_type>_model`: Defines the neural network architecture.

Supported Models:
1.  **Autoencoder (Content-Based)**: Reconstructs item vectors to find similarities.
2.  **Matrix Factorization (Collaborative)**: Learns user/item embeddings to predict ratings.
3.  **TensorFlow Recommenders (TFRS) (Retrieval)**: A Two-Tower model for efficient retrieval
    using user and item metadata.


Model Training Module

This script defines the ModelTrainingService class, which orchestrates the training,
logging, and artifact saving for three types of recommendation models:
Autoencoder, Matrix Factorization, and TFRS.

Logic of Operation:
1.  **Configuration**: Loads model hyperparameters and training settings (epochs, batch size)
    from `configs/model_params.yaml`.
2.  **Model Building**: Uses `ModelBuilder` to construct the specific model architecture (AE, MF, or TFRS).
3.  **Data Preparation**: Loads transformed data and uses `ModelBuilder` to prepare it for the specific model.
4.  **Training**: Runs the training loop validation data.
5.  **Logging**: Logs parameters and metrics to MLflow for experiment tracking.
6.  **Persistence**: Saves the trained models (H5 or SavedModel) and necessary context artifacts
    (encoders, vocabularies) to the filesystem for later use in prediction.


Model Evaluation Module

This script defines the ModelEvaluationService class, which provides a comprehensive framework
for evaluating the performance of trained recommendation models (Autoencoder, MF, and TFRS).

Logic of Operation:
1.  **Metric Calculation**: Implements ranking metrics (Precision@K, MAP@K, NDCG@K) to assess
    recommendation quality.
2.  **Ranking Model Evaluation (MF & TFRS)**:
    - Loads test data and the trained model.
    - Generates predictions for a sample of users against a set of candidate items (excluding already played).
    - Compares top-K recommendations against the actual ground-truth items in the test set.
    - Logs aggregated metrics to MLflow.
3.  **Autoencoder Evaluation**:
    - Uses a content-based "Categorized Accuracy" metric.
    - Finds nearest neighbors of test items using the learned embeddings.
    - Checks if the neighbors share similar genres/tags with the query item.
4.  **Reporting**: Logs all calculated metrics to MLflow for comparison across experiments.

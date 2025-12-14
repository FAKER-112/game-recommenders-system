Training Pipeline Module

This script defines the TrainingPipeline class, which orchestrates the end-to-end model training workflow.
It integrates data ingestion, cleaning, feature engineering, and model training into a single execution flow.

Logic of Operation:
1.  **Environment Setup**: Configures TensorFlow environment variables (legacy Keras, CPU/GPU settings).
2.  **Configuration**: Loads pipeline parameters from `configs/pipeline_params.yaml`.
3.  **Pipeline Execution (`train_model` method)**:
    - **Data Ingestion**: Downloads raw data using `LoadDataService`.
    - **Data Cleaning**: Processes and merges raw data using `CleanDataService`.
    - **Feature Engineering**: Transforms cleaned data into model-ready features using `FeatureEngineeringService`.
    - **Model Training**: Trains the specified model (Autoencoder, MF, or TFRS) using `ModelTrainingService`.

Prediction Pipeline Module

This script defines the PredictionPipeline class, which serves as the inference engine
for the recommender system. It handles loading trained models and generating recommendations
for users or finding similar items.

Logic of Operation:
1.  **Initialization**:
    - Loads the specified model type ('tfrs', 'mf', or 'autoencoder') based on configuration.
    - Loads the corresponding trained model artifacts (H5 files, SavedModels).
    - Loads necessary context artifacts (encoders, vocabularies, candidate lists) required
      to map raw inputs to model inputs and model outputs back to human-readable names.
2.  **Recommendation (`recommend` method)**:
    - Accepts a User ID.
    - Uses the loaded model (or retrieval index) to predict/retrieve top-N items.
    - Maps internal IDs back to Item Names.
3.  **Similarity Search (`get_similar_items` method)**:
    - Accepts an Item Name.
    - Uses item embeddings (Autoencoder) or item-to-item indices (TFRS) to find closest neighbors.
    - Returns a list of similar items.

"""
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
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
import mlflow.sklearn
from pathlib import Path
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.utils import load_config
from src.models.build_model import ModelBuilder


class ModelTrainingService:
    def __init__(self, model_params_yaml: str = str(Path("configs/model_params.yaml"))):
        try:
            self.config = load_config(model_params_yaml)
            self.logger = logger

            # Load model training config
            model_cfg = self.config.get("model_training", {})
            self.root_dir = model_cfg.get("root_dir", "artifacts/models")
            self.context_dir = model_cfg.get("context_dir", "articfact/context")
            os.makedirs(self.root_dir, exist_ok=True)
            os.makedirs(self.context_dir, exist_ok=True)
            self.transformed_train_path = model_cfg.get(
                "transformed_train_path", "data/processed/train.csv"
            )
            self.transformed_test_path = model_cfg.get(
                "transformed_test_path", "data/processed/test.csv"
            )
            self.transformed_data_path = model_cfg.get(
                "transformed_data_path", "data/processed/data.csv"
            )
            self.model_builder = ModelBuilder()

            # Initialize MLflow
            mlflow.set_experiment("Game_Recommender_System")

        except Exception as e:
            raise CustomException(e)

    def train_autoencoder(self):
        try:
            self.logger.info("Starting Autoencoder training...")
            with mlflow.start_run(run_name="Autoencoder"):
                # Load Data
                train_df = pd.read_csv(self.transformed_train_path)
                test_df = pd.read_csv(self.transformed_test_path)
                df = pd.read_csv(self.transformed_data_path)

                # Prepare Data
                X, indices, itemlist, input_dim = (
                    self.model_builder.prepare_data_autoencoder(df)
                )

                np.save(os.path.join(self.context_dir, "autoencoder_X.npy"), X)
                indices.to_csv(
                    os.path.join(self.context_dir, "autoencoder_indices.csv")
                )
                listpath = os.path.join(self.context_dir, "autoencoder_itemlist.json")
                with open(listpath, "w") as f:
                    json.dump(itemlist, f, indent=4)
                # Build Model
                params = self.config["model_params"]["autoencoder"]
                mlflow.log_params(params)

                autoencoder = self.model_builder.build_autoencoder_model(
                    input_dim=input_dim, encoding_dim=params.get("encoding_dim", 64)
                )

                # Train
                history = autoencoder.fit(
                    X,
                    X,
                    epochs=self.config["model_training"]["epochs"],
                    batch_size=self.config["model_training"]["batch_size"],
                    verbose=1,
                )

                # Log Metrics
                for metric, values in history.history.items():
                    mlflow.log_metric(metric, values[-1])

                # Save Model
                model_path = os.path.join(self.root_dir, "autoencoder.h5")
                autoencoder.save(model_path)
                mlflow.keras.log_model(autoencoder, "model")
                self.logger.info(f"Autoencoder model saved to {model_path}")

        except Exception as e:
            raise CustomException(e)

    def train_matrix_factorization(self):
        try:
            self.logger.info("Starting Matrix Factorization training...")
            with mlflow.start_run(run_name="Matrix_Factorization"):
                # Load Data
                train_df = pd.read_csv(self.transformed_train_path)
                test_df = pd.read_csv(self.transformed_test_path)

                # Prepare Data
                (
                    train_user_ids,
                    train_item_ids,
                    test_user_ids,
                    test_item_ids,
                    num_users,
                    num_items,
                    user_encoder,
                    item_encoder,
                ) = self.model_builder.prepare_data_mf(train_df, test_df)
                user_encoder_path = os.path.join(
                    self.context_dir, "mf_user_encoder.pkl"
                )
                with open(user_encoder_path, "wb") as f:
                    pickle.dump(user_encoder, f)

                item_encoder_path = os.path.join(
                    self.context_dir, "mf_item_encoder.pkl"
                )
                with open(item_encoder_path, "wb") as f:
                    pickle.dump(item_encoder, f)
                # Build Model
                params = self.config["model_params"]["matrix_factorization"]
                mlflow.log_params(params)

                mf_model = self.model_builder.build_mf_model(
                    num_users=num_users,
                    num_items=num_items,
                    embedding_size=params.get("embedding_size", 50),
                )

                # Train
                history = mf_model.fit(
                    [train_user_ids, train_item_ids],
                    train_df["rating"],
                    epochs=self.config["model_training"]["epochs"],
                    batch_size=self.config["model_training"]["batch_size"],
                    validation_data=([test_user_ids, test_item_ids], test_df["rating"]),
                    verbose=1,
                )

                # Log Metrics
                for metric, values in history.history.items():
                    mlflow.log_metric(metric, values[-1])

                # Save Model
                model_path = os.path.join(self.root_dir, "mf_model.h5")
                mf_model.save(model_path)
                mlflow.keras.log_model(mf_model, "model")
                self.logger.info(f"Matrix Factorization model saved to {model_path}")

        except Exception as e:
            raise CustomException(e)

    def train_tfrs(self):
        try:
            self.logger.info("Starting TFRS training...")
            with mlflow.start_run(run_name="TFRS_Retrieval"):
                # Load Data
                train_df = pd.read_csv(self.transformed_train_path)
                test_df = pd.read_csv(self.transformed_test_path)
                # train_df = train_df[:10000]
                # test_df = test_df[:1000]

                # Prepare Data
                data_dict = self.model_builder.prepare_data_tfrs(train_df, test_df)

                # Build Model
                params = self.config["model_params"]["tfrs"]
                mlflow.log_params(params)

                # Save context data needed for inference
                context_data = {
                    "unique_items_df": data_dict["unique_items_df"],
                    "user_vocab_size": data_dict[
                        "user_ids_vocabulary"
                    ].vocabulary_size(),
                    "item_vocab_size": data_dict[
                        "item_titles_vocabulary"
                    ].vocabulary_size(),
                    "text_vocab_size": data_dict["text_vectorizer"].vocabulary_size(),
                }

                # Save unique items dataframe
                candidates_path = os.path.join(self.context_dir, "tfrs_candidates.csv")
                data_dict["unique_items_df"].to_csv(candidates_path, index=False)

                # Save vocabularies
                user_vocab_path = os.path.join(self.context_dir, "user_vocabulary.pkl")
                with open(user_vocab_path, "wb") as f:
                    pickle.dump(data_dict["user_ids_vocabulary"].get_vocabulary(), f)

                item_vocab_path = os.path.join(self.context_dir, "item_vocabulary.pkl")
                with open(item_vocab_path, "wb") as f:
                    pickle.dump(data_dict["item_titles_vocabulary"].get_vocabulary(), f)

                text_vocab_path = os.path.join(self.context_dir, "text_vocabulary.pkl")
                text_vocab_config = data_dict["text_vectorizer"].get_config()
                text_weights = data_dict["text_vectorizer"].get_weights()
                with open(text_vocab_path, "wb") as f:
                    pickle.dump(
                        {"config": text_vocab_config, "weights": text_weights}, f
                    )

                # Save context metadata
                context_metadata_path = os.path.join(
                    self.context_dir, "tfrs_metadata.json"
                )
                with open(context_metadata_path, "w") as f:
                    json.dump(context_data, f, indent=4, default=str)

                # Build and compile model
                tfrs_model = self.model_builder.build_tfrs_model(data_dict)
                tfrs_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
                )

                # Train
                history = tfrs_model.fit(
                    data_dict["train_ds"].batch(
                        self.config["model_training"]["batch_size"]
                    ),
                    epochs=2,
                    validation_data=data_dict["test_ds"].batch(
                        self.config["model_training"]["batch_size"]
                    ),
                    verbose=1,
                )

                # Log Metrics
                for metric, values in history.history.items():
                    mlflow.log_metric(metric, values[-1])

                # ===== SAVE MODEL COMPONENTS =====

                # 1. Save user model
                user_model_path = os.path.join(self.root_dir, "tfrs_user_model")
                tfrs_model.user_model.save(user_model_path)
                self.logger.info(f"User model saved to {user_model_path}")

                # 2. Save item model
                item_model_path = os.path.join(self.root_dir, "tfrs_item_model")
                tfrs_model.item_model.save(item_model_path)
                self.logger.info(f"Item model saved to {item_model_path}")

                # 3. Create and save USER-TO-ITEM retrieval index for fast inference
                index_user = tfrs.layers.factorized_top_k.BruteForce(
                    tfrs_model.user_model
                )
                index_user.index_from_dataset(
                    data_dict["items"]
                    .batch(128)
                    .map(lambda x: (x["item_name"], tfrs_model.item_model(x)))
                )

                # CRITICAL: Call the index at least once to build it before saving
                sample_user = tf.constant([str(train_df["user_id"].iloc[0])])
                _ = index_user(sample_user)

                # Save using tf.saved_model.save
                index_user_path = os.path.join(self.root_dir, "tfrs_retrieval_index")
                tf.saved_model.save(index_user, index_user_path)
                self.logger.info(
                    f"User-to-item retrieval index saved to {index_user_path}"
                )

                # 4. Create and save ITEM-TO-ITEM retrieval index for similar items
                item_to_item_index = tfrs.layers.factorized_top_k.BruteForce(
                    tfrs_model.item_model
                )
                item_to_item_index.index_from_dataset(
                    data_dict["items"]
                    .batch(128)
                    .map(lambda x: (x["item_name"], tfrs_model.item_model(x)))
                )

                # Call with sample item (must be dict format for item_model)
                sample_item_name = str(train_df["item_name"].iloc[0])
                sample_item_text = str(train_df["item_text"].iloc[0])
                sample_item_input = {
                    "item_name": tf.constant([sample_item_name]),
                    "item_text": tf.constant([sample_item_text]),
                }
                _ = item_to_item_index(sample_item_input)

                # Save item-to-item index
                item_index_path = os.path.join(self.root_dir, "tfrs_item_index")
                tf.saved_model.save(item_to_item_index, item_index_path)
                self.logger.info(
                    f"Item-to-item retrieval index saved to {item_index_path}"
                )

                # 5. Save full model weights as backup
                weights_path = os.path.join(self.root_dir, "tfrs_weights")
                tfrs_model.save_weights(weights_path)
                self.logger.info(f"Model weights saved to {weights_path}")

                # Log artifacts to MLflow
                mlflow.log_artifacts(self.context_dir, artifact_path="context")
                mlflow.log_artifacts(self.root_dir, artifact_path="models")

                self.logger.info(
                    "TFRS model training and saving completed successfully"
                )

                return {
                    "user_model_path": user_model_path,
                    "item_model_path": item_model_path,
                    "index_user_path": index_user_path,
                    "item_index_path": item_index_path,
                    "weights_path": weights_path,
                }

        except Exception as e:
            raise CustomException(e)

    def run(self, model_name: str):
        try:
            if model_name == "autoencoder":
                self.train_autoencoder()
            elif model_name == "mf":
                self.train_matrix_factorization()
            elif model_name == "tfrs":
                self.train_tfrs()
            elif model_name == "all":
                self.train_autoencoder()
                self.train_matrix_factorization()
                self.train_tfrs()
            else:
                raise ValueError(f"Invalid model name: {model_name}")

        except Exception as e:
            raise CustomException(e)

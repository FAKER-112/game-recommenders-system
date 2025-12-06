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
            self.context_dir= model_cfg.get('context_dir', 'articfact/context')
            os.makedirs(self.root_dir, exist_ok=True)
            os.makedirs(self.context_dir, exist_ok=True)
            self.transformed_train_path = model_cfg.get("transformed_train_path", 'data/processed/train.csv')
            self.transformed_test_path = model_cfg.get("transformed_test_path", 'data/processed/test.csv')
            self.transformed_data_path = model_cfg.get("transformed_data_path", 'data/processed/data.csv')
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
                df= pd.read_csv(self.transformed_data_path)

                # Prepare Data
                X, indices, itemlist, input_dim = (
                    self.model_builder.prepare_data_autoencoder(df)
                )

                np.save(os.path.join(self.context_dir,'autoencoder_X.npy'), X)
                indices.to_csv(os.path.join(self.context_dir,'autoencoder_indices.csv'))
                listpath= os.path.join(self.context_dir,'autoencoder_itemlist.json')
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
                    item_encoder
                ) = self.model_builder.prepare_data_mf(train_df, test_df)
                user_encoder_path = os.path.join(self.context_dir, "mf_user_encoder.pkl")
                with open(user_encoder_path, "wb") as f:
                    pickle.dump(user_encoder, f)

                item_encoder_path = os.path.join(self.context_dir, "mf_item_encoder.pkl")
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
                train_df = train_df[:10000]
                test_df= test_df[:1000]
                # Prepare Data
                data_dict = self.model_builder.prepare_data_tfrs(train_df, test_df)

                # Build Model
                params = self.config["model_params"]["tfrs"]
                mlflow.log_params(params)
                candidates_path = os.path.join(self.context_dir, "tfrs_candidates.pkl")
                
                with open(candidates_path, "wb") as f:
                    pickle.dump(data_dict["unique_items_df"], f)
                tfrs_model = self.model_builder.build_tfrs_model(data_dict)
                tfrs_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1)
                )

                # Train
                # TFRS models expect dictionary inputs
                history = tfrs_model.fit(
                    data_dict["train_ds"].batch(
                        self.config["model_training"]["batch_size"]
                    ),
                    epochs=self.config["model_training"]["epochs"],
                    validation_data=data_dict["test_ds"].batch(
                        self.config["model_training"]["batch_size"]
                    ),
                    verbose=1,
                )

                # Log Metrics
                for metric, values in history.history.items():
                    mlflow.log_metric(metric, values[-1])

                # Save Model (TFRS models are often saved as SavedModel format)
                model_path = os.path.join(self.root_dir, "tfrs_model")
                tf.saved_model.save(tfrs_model, model_path)
                mlflow.keras.log_model(tfrs_model, "model")
                self.logger.info(f"TFRS model saved to {model_path}")

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

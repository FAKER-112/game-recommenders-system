import numpy as np
import pandas as pd
import math
import os
import sys
import ast
import tensorflow as tf
import mlflow
import mlflow.keras
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.utils import load_config, calculate_precision_at_k, calculate_ap_at_k, calculate_ndcg_at_k
from src.models.build_model import ModelBuilder

class ModelEvaluationService:
    '''
    This class is used to evaluate the models.
    '''
    def __init__(self, config_path: str = "configs/config.yaml"):
        '''
        This function is used to initialize the ModelEvaluationService class.
        '''
        try:
            self.config = load_config(config_path)
            self.logger = logger

            model_cfg = self.config.get("model_training", {})
            self.root_dir = model_cfg.get("root_dir", "models")
            self.transformed_train_path = model_cfg.get("transformed_train_path")
            self.transformed_test_path = model_cfg.get("transformed_test_path")
            self.transformed_data_path = model_cfg.get("transformed_data_path")

            feat_cfg = self.config.get("feature_engineering", {})
            self.cleaned_data_path = feat_cfg.get("cleaned_data_path")

            self.model_builder = ModelBuilder()

            # Initialize MLflow
            mlflow.set_experiment("Game_Recommender_System")

        except Exception as e:
            raise CustomException(e)

    def _evaluate_ranking_model(
        self, predict_fn, user_ids_to_test, train_df, test_df, all_item_ids, k=10
    ):
        """
        Generic evaluation loop for ranking models.
        Args:
            predict_fn: A function that takes (user_id, candidate_ids) and returns scores.
        """
        precisions = []
        maps = []
        ndcgs = []

        test_user_groups = test_df.groupby("user_id")
        train_user_groups = train_df.groupby("user_id")

        count = 0
        total_users = len(user_ids_to_test)

        for user_id in user_ids_to_test:
            count += 1
            if count % 50 == 0:
                self.logger.info(f"Evaluated {count}/{total_users} users...")

            if user_id not in test_user_groups.groups:
                continue

            user_test_data = test_user_groups.get_group(user_id)
            relevant_items = set(
                user_test_data[user_test_data["rating"] > 0]["item_id"].values
            )

            if len(relevant_items) == 0:
                continue

            played_in_train = set()
            if user_id in train_user_groups.groups:
                played_in_train = set(
                    train_user_groups.get_group(user_id)["item_id"].values
                )

            candidates = np.setdiff1d(all_item_ids, list(played_in_train))

            # Predict using the passed function
            predictions = predict_fn(user_id, candidates)

            # Top-K
            top_indices = predictions.argsort()[-k:][::-1]
            recommended_ids = candidates[top_indices]

            # Metrics
            precisions.append(
                calculate_precision_at_k(recommended_ids, relevant_items, k)
            )
            maps.append(calculate_ap_at_k(recommended_ids, relevant_items, k))
            ndcgs.append(calculate_ndcg_at_k(recommended_ids, relevant_items, k))

        return {
            f"Precision@{k}": np.mean(precisions) if precisions else 0.0,
            f"MAP@{k}": np.mean(maps) if maps else 0.0,
            f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0,
        }

    def evaluate_mf_model(self, k=10, sample_size=100):
        try:
            self.logger.info("Starting Matrix Factorization Evaluation...")

            train_df_raw = pd.read_csv(self.transformed_train_path)
            test_df_raw = pd.read_csv(self.transformed_test_path)

            train_user_ids, train_item_ids, test_user_ids, test_item_ids, _, _ = (
                self.model_builder.prepare_data_mf(train_df_raw, test_df_raw)
            )

            train_df_encoded = pd.DataFrame(
                {
                    "user_id": train_user_ids,
                    "item_id": train_item_ids,
                    "rating": train_df_raw["rating"],
                }
            )
            test_df_encoded = pd.DataFrame(
                {
                    "user_id": test_user_ids,
                    "item_id": test_item_ids,
                    "rating": test_df_raw["rating"],
                }
            )

            all_item_ids = np.unique(np.concatenate([train_item_ids, test_item_ids]))

            model_path = os.path.join(self.root_dir, "mf_model.h5")
            if not os.path.exists(model_path):
                self.logger.warning(f"Model not found at {model_path}. Skipping.")
                return

            model = tf.keras.models.load_model(model_path)

            # Define prediction function for MF
            def mf_predict(user_id, candidates):
                user_input = np.array([user_id] * len(candidates))
                return model.predict([user_input, candidates], verbose=0).flatten()

            unique_test_users = test_df_encoded["user_id"].unique()
            sample_users = np.random.choice(
                unique_test_users,
                min(sample_size, len(unique_test_users)),
                replace=False,
            )

            metrics = self._evaluate_ranking_model(
                mf_predict,
                sample_users,
                train_df_encoded,
                test_df_encoded,
                all_item_ids,
                k=k,
            )

            self.logger.info(f"MF Evaluation Results: {metrics}")
            with mlflow.start_run(run_name="MF_Evaluation"):
                mlflow.log_metrics(metrics)

        except Exception as e:
            raise CustomException(e)

    def evaluate_autoencoder_model(self, k=5, sample_size=100):
        '''
        This function is used to evaluate the Autoencoder model.
        '''
        try:
            self.logger.info("Starting Autoencoder Evaluation...")

            train_df = pd.read_csv(self.transformed_train_path)
            test_df = pd.read_csv(self.transformed_test_path)

            if os.path.exists(self.cleaned_data_path):
                full_df = pd.read_csv(self.cleaned_data_path)
                item_genres_map = full_df.set_index("item_name")["genres"].to_dict()
            else:
                self.logger.warning(
                    "Cleaned data not found. Cannot compute Categorized Accuracy."
                )
                item_genres_map = {}

            all_items_df = (
                pd.concat([train_df, test_df])
                .drop_duplicates(subset="item_name")
                .reset_index(drop=True)
            )

            from sklearn.feature_extraction.text import TfidfVectorizer

            tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
            train_content = train_df.drop_duplicates(subset="item_name")
            tfidf.fit(train_content["item_text"].fillna(""))

            X_all = tfidf.transform(all_items_df["item_text"].fillna("")).toarray()

            model_path = os.path.join(self.root_dir, "autoencoder.h5")
            if not os.path.exists(model_path):
                self.logger.warning("Autoencoder model not found. Skipping.")
                return

            autoencoder = tf.keras.models.load_model(model_path)
            encoder = tf.keras.Model(
                inputs=autoencoder.input,
                outputs=autoencoder.get_layer("bottleneck").output,
            )

            embeddings = encoder.predict(X_all, verbose=0)

            nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(embeddings)
            distances, indices = nbrs.kneighbors(embeddings)

            accuracies = []
            sample_indices = np.random.choice(
                len(all_items_df), min(sample_size, len(all_items_df)), replace=False
            )

            for idx in sample_indices:
                query_item_name = all_items_df.iloc[idx]["item_name"]
                query_genres = item_genres_map.get(query_item_name, "[]")
                try:
                    query_genres = (
                        set(ast.literal_eval(query_genres))
                        if isinstance(query_genres, str)
                        else set()
                    )
                except:
                    query_genres = set()

                if not query_genres:
                    continue

                neighbor_indices = indices[idx][1:]
                match_count = 0

                for n_idx in neighbor_indices:
                    neighbor_name = all_items_df.iloc[n_idx]["item_name"]
                    neighbor_genres = item_genres_map.get(neighbor_name, "[]")
                    try:
                        neighbor_genres = (
                            set(ast.literal_eval(neighbor_genres))
                            if isinstance(neighbor_genres, str)
                            else set()
                        )
                    except:
                        neighbor_genres = set()

                    if not query_genres.isdisjoint(neighbor_genres):
                        match_count += 1

                accuracies.append(match_count / k)

            avg_cat_accuracy = np.mean(accuracies) if accuracies else 0.0

            metrics = {f"CategorizedAccuracy@{k}": avg_cat_accuracy}
            self.logger.info(f"Autoencoder Evaluation Results: {metrics}")

            with mlflow.start_run(run_name="Autoencoder_Evaluation"):
                mlflow.log_metrics(metrics)

        except Exception as e:
            raise CustomException(e)

    def evaluate_tfrs_model(self, k=10, sample_size=100):
        '''
        This function is used to evaluate the TFRS model.
        '''
        try:
            self.logger.info("Starting TFRS Evaluation (Brute Force)...")

            # 1. Load Data
            train_df = pd.read_csv(self.transformed_train_path)
            test_df = pd.read_csv(self.transformed_test_path)

            # Ensure strings
            train_df["user_id"] = train_df["user_id"].astype(str)
            test_df["user_id"] = test_df["user_id"].astype(str)
            train_df["item_name"] = train_df["item_name"].astype(str)
            test_df["item_name"] = test_df["item_name"].astype(str)

            # 2. Load Model
            model_path = os.path.join(self.root_dir, "tfrs_model")
            if not os.path.exists(model_path):
                self.logger.warning("TFRS model not found. Skipping.")
                return

            tfrs_model = tf.saved_model.load(model_path)

            # 3. Prepare Item Map (ID -> Text)
            # TFRS Item Tower needs (item_name, item_text)
            all_items_df = (
                pd.concat([train_df, test_df])
                .drop_duplicates(subset="item_name")
                .set_index("item_name")
            )
            all_item_ids = all_items_df.index.values

            # 4. Define Prediction Function
            def tfrs_predict(user_id, candidates):
                # User Embedding
                user_emb = tfrs_model.user_model(np.array([str(user_id)]))

                # Candidate Embeddings
                # Note: For large candidate sets, this should be batched or pre-computed
                candidate_texts = (
                    all_items_df.loc[candidates]["item_text"].fillna("").values
                )
                item_embs = tfrs_model.item_model(
                    np.array(candidates), np.array(candidate_texts)
                )

                # Scores = Dot Product
                scores = tf.matmul(user_emb, item_embs, transpose_b=True)
                return scores.numpy().flatten()

            # 5. Evaluate
            # Rename columns to match generic evaluator expectation (item_name -> item_id)
            train_df_renamed = train_df.rename(columns={"item_name": "item_id"})
            test_df_renamed = test_df.rename(columns={"item_name": "item_id"})

            unique_test_users = test_df_renamed["user_id"].unique()
            sample_users = np.random.choice(
                unique_test_users,
                min(sample_size, len(unique_test_users)),
                replace=False,
            )

            metrics = self._evaluate_ranking_model(
                tfrs_predict,
                sample_users,
                train_df_renamed,
                test_df_renamed,
                all_item_ids,
                k=k,
            )

            self.logger.info(f"TFRS Evaluation Results: {metrics}")
            with mlflow.start_run(run_name="TFRS_Evaluation"):
                mlflow.log_metrics(metrics)

        except Exception as e:
            self.logger.error(f"TFRS Evaluation failed: {e}")
            raise CustomException(e)

    def run(self, model_name: str):
        try:
            if model_name == "mf":
                self.evaluate_mf_model()
            elif model_name == "autoencoder":
                self.evaluate_autoencoder_model()
            elif model_name == "tfrs":
                self.evaluate_tfrs_model()
            else:
                raise ValueError(f"Invalid model name: {model_name}")
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    evaluator = ModelEvaluationService()
    evaluator.evaluate_all_models()

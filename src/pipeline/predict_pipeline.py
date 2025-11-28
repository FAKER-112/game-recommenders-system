import os
import pickle
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.utils import load_config


class PredictionPipeline:
    def __init__(
        self, model_name: str = "tfrs", config_path: str = "configs/config.yaml"
    ):
        """
        Initialize the PredictionPipeline.
        Args:
            model_name: 'tfrs', 'mf', or 'autoencoder'
            config_path: Path to configuration file
        """
        try:
            self.config = load_config(config_path)
            self.model_name = model_name
            self.logger = logger

            # Load paths from config
            model_cfg = self.config.get("model_training", {})
            self.root_dir = model_cfg.get("root_dir", "models")
            self.context_dir = model_cfg.get(
                "context_dir", "models/context"
            )  # Assuming context dir for pickles

            # Ensure context dir exists if we are looking for files there
            if not os.path.exists(self.context_dir):
                # Fallback to root_dir if context_dir not explicitly set or doesn't exist
                self.context_dir = self.root_dir

            self._load_model()
            self._load_artifacts()

        except Exception as e:
            raise CustomException(e)

    def _load_model(self):
        """Loads the specified model."""
        try:
            model_path = os.path.join(self.root_dir, self.model_name)
            if self.model_name == "mf":
                self.model = load_model(f"{model_path}.h5")
            elif self.model_name == "autoencoder":
                self.model = load_model(f"{model_path}.h5")
            elif self.model_name == "tfrs":
                self.model = tf.saved_model.load(model_path)
            else:
                raise ValueError(f"Invalid model name: {self.model_name}")
            self.logger.info(f"Loaded {self.model_name} model successfully.")
        except Exception as e:
            raise CustomException(e)

    def _load_artifacts(self):
        """Loads necessary artifacts (encoders, indices, etc.) based on model type."""
        try:
            if self.model_name == "autoencoder":
                # Load pre-computed embeddings or data to compute them
                # For efficiency, we assume X and indices are saved.
                # If not, we might need to re-compute or load from a specific path.
                # Based on user snippet, loading from specific files.
                # Ideally these paths should be in config.

                # Check if files exist, otherwise log warning
                if os.path.exists("autoencoder_X.npz"):
                    self.X = np.load("autoencoder_X.npz")[
                        "arr_0"
                    ]  # Assuming saved with save_npz
                else:
                    self.logger.warning("autoencoder_X.npz not found.")
                    self.X = None

                if os.path.exists("autoencoder_indices.csv"):
                    self.indices = pd.read_csv(
                        "autoencoder_indices.csv", index_col="item_name"
                    )["index"]
                else:
                    self.logger.warning("autoencoder_indices.csv not found.")
                    self.indices = None

                if os.path.exists("autoencoder_itemlist.json"):
                    with open("autoencoder_itemlist.json", "r") as f:
                        self.global_item_names = json.load(f)
                else:
                    self.global_item_names = []

            elif self.model_name == "mf":
                # Load encoders
                with open(
                    os.path.join(self.context_dir, "mf_user_encoder.pkl"), "rb"
                ) as f:
                    self.user_encoder = pickle.load(f)
                with open(
                    os.path.join(self.context_dir, "mf_item_encoder.pkl"), "rb"
                ) as f:
                    self.item_encoder = pickle.load(f)

            elif self.model_name == "tfrs":
                # Load candidates for retrieval
                candidates_path = os.path.join(self.context_dir, "tfrs_candidates.pkl")
                if os.path.exists(candidates_path):
                    with open(candidates_path, "rb") as f:
                        self.unique_items_df = pickle.load(f)
                else:
                    self.logger.warning(
                        f"Candidates file not found at {candidates_path}"
                    )
                    self.unique_items_df = None

        except Exception as e:
            self.logger.error(f"Error loading artifacts: {e}")
            # Don't raise here to allow partial initialization if needed, or raise if critical.
            # For now, we'll raise to be safe.
            raise CustomException(e)

    def recommend(self, user_id, n_rec=10):
        """
        Generates recommendations for a user.
        """
        try:
            if self.model_name == "tfrs":
                return self._recommend_tfrs(user_id, n_rec)
            elif self.model_name == "mf":
                return self._recommend_mf(user_id, n_rec)
            else:
                self.logger.warning(f"Recommend not implemented for {self.model_name}")
                return []
        except Exception as e:
            raise CustomException(e)

    def _recommend_mf(self, user_id, n_rec):
        if user_id not in self.user_encoder.classes_:
            self.logger.warning(f"User {user_id} not found in training data.")
            return []

        encoded_user_id = self.user_encoder.transform([user_id])[0]
        num_items = len(self.item_encoder.classes_)
        all_item_ids = np.arange(num_items)

        user_input = np.full(num_items, encoded_user_id)
        item_input = all_item_ids

        predictions = self.model.predict([user_input, item_input], verbose=0).flatten()
        top_indices = predictions.argsort()[-n_rec:][::-1]

        recommended_item_ids = all_item_ids[top_indices]
        recommended_items = self.item_encoder.inverse_transform(recommended_item_ids)
        return recommended_items.tolist()

    def _recommend_tfrs(self, user_id, n_rec):
        if self.unique_items_df is None:
            raise ValueError("TFRS candidates not loaded.")

        # Create BruteForce layer for retrieval
        # Note: In a real production setting, we would serve the index.
        # Here we rebuild it on the fly or use the brute force approach.
        # For efficiency, we'll use the BruteForce layer from TFRS.

        index = tfrs.layers.factorized_top_k.BruteForce(self.model.user_model)

        # Convert candidates to dataset
        items_ds = tf.data.Dataset.from_tensor_slices(
            dict(self.unique_items_df[["item_name", "item_text"]])
        )

        # Index the candidates
        index.index_from_dataset(
            tf.data.Dataset.zip(
                (
                    items_ds.map(lambda x: x["item_name"]),
                    items_ds.batch(128).map(
                        lambda x: self.model.item_model(x["item_name"], x["item_text"])
                    ),
                )
            )
        )

        # Get recommendations
        _, titles = index(tf.constant([str(user_id)]))
        recommended_titles = [t.decode("utf-8") for t in titles[0, :n_rec].numpy()]
        return recommended_titles

    def get_similar_items(self, item_name: str, k=10) -> list:
        """
        Finds similar items using Autoencoder embeddings.
        """
        try:
            if self.model_name != "autoencoder":
                self.logger.warning(
                    "get_similar_items is only available for Autoencoder."
                )
                return []

            if self.X is None or self.indices is None:
                raise ValueError("Autoencoder artifacts (X, indices) not loaded.")

            # Extract encoder part
            encoder = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("bottleneck").output,
            )

            # Compute embeddings for all items (if not already cached/loaded)
            # In production, these should be pre-computed.
            # Here we assume self.X is the raw input features, so we predict.
            embeddings = encoder.predict(self.X, verbose=0)

            # Calculate similarity
            # Note: Computing full cosine similarity matrix is expensive (N x N).
            # Better to compute only for the query item against all.

            if item_name not in self.indices:
                return ["Item not found."]

            idx = self.indices[item_name]
            query_embedding = embeddings[idx].reshape(1, -1)

            # Compute cosine similarity between query and all others
            sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

            # Get top K
            top_indices = sim_scores.argsort()[-(k + 1) :][
                ::-1
            ]  # +1 because self is included

            recommendations = []
            for i in top_indices:
                if i != idx:  # Exclude self
                    recommendations.append(self.global_item_names[i])
                    if len(recommendations) == k:
                        break

            return recommendations

        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    # Example Usage
    try:
        # TFRS
        pipeline_tfrs = PredictionPipeline(model_name="tfrs")
        # print(pipeline_tfrs.recommend(user_id="76561197970982479", n_rec=5))

        # MF
        # pipeline_mf = PredictionPipeline(model_name='mf')
        # print(pipeline_mf.recommend(user_id="76561197970982479", n_rec=5))

        # Autoencoder
        # pipeline_ae = PredictionPipeline(model_name='autoencoder')
        # print(pipeline_ae.get_similar_items(item_name="Counter-Strike", k=5))

    except Exception as e:
        print(e)

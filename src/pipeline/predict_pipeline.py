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
        self, model_name: str = "tfrs", config_path: str = "configs/model_params.yaml"
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
            self.root_dir = model_cfg.get("root_dir", "artifacts/models")
            self.context_dir = model_cfg.get("context_dir", "artifacts/context")

            self._load_model()
            self._load_artifacts()

        except Exception as e:
            raise CustomException(e)

    def _load_model(self):
        """Loads the specified model."""
        try:
            if self.model_name == "mf":
                model_path = os.path.join(self.root_dir, "mf_model.h5")
                self.model = load_model(model_path)
                self.logger.info(f"Loaded MF model from {model_path}")
                
            elif self.model_name == "autoencoder":
                model_path = os.path.join(self.root_dir, "autoencoder.h5")
                self.model = load_model(model_path)
                self.logger.info(f"Loaded Autoencoder model from {model_path}")
                
            elif self.model_name == "tfrs":
                # Load the pre-built retrieval index (user-to-item)
                index_path = os.path.join(self.root_dir, "tfrs_retrieval_index")
                if os.path.exists(index_path):
                    self.retrieval_index = tf.saved_model.load(index_path)
                    self.logger.info(f"Loaded TFRS retrieval index from {index_path}")
                else:
                    raise FileNotFoundError(f"TFRS retrieval index not found at {index_path}")
                
                # Load item-to-item index for similar items
                item_index_path = os.path.join(self.root_dir, "tfrs_item_index")
                if os.path.exists(item_index_path):
                    self.item_index = tf.saved_model.load(item_index_path)
                    self.logger.info(f"Loaded TFRS item index from {item_index_path}")
                else:
                    self.logger.warning(f"TFRS item index not found at {item_index_path}")
                    self.item_index = None
            else:
                raise ValueError(f"Invalid model name: {self.model_name}")
                
        except Exception as e:
            raise CustomException(e)

    def _load_artifacts(self):
        """Loads necessary artifacts (encoders, indices, etc.) based on model type."""
        try:
            if self.model_name == "autoencoder":
                # Load pre-computed embeddings and indices
                X_path = os.path.join(self.context_dir, "autoencoder_X.npy")
                indices_path = os.path.join(self.context_dir, "autoencoder_indices.csv")
                itemlist_path = os.path.join(self.context_dir, "autoencoder_itemlist.json")
                
                if os.path.exists(X_path):
                    self.X = np.load(X_path)
                    self.logger.info(f"Loaded autoencoder X from {X_path}")
                else:
                    self.logger.warning(f"autoencoder_X.npy not found at {X_path}")
                    self.X = None

                if os.path.exists(indices_path):
                    indices_df = pd.read_csv(indices_path)
                    # Create Series with item_name as index
                    self.indices = pd.Series(
                        indices_df.iloc[:, 1].values, 
                        index=indices_df.iloc[:, 0].values
                    )
                    self.logger.info(f"Loaded autoencoder indices from {indices_path}")
                else:
                    self.logger.warning(f"autoencoder indices not found at {indices_path}")
                    self.indices = None

                if os.path.exists(itemlist_path):
                    with open(itemlist_path, "r") as f:
                        self.global_item_names = json.load(f)
                    self.logger.info(f"Loaded item list from {itemlist_path}")
                else:
                    self.logger.warning(f"Item list not found at {itemlist_path}")
                    self.global_item_names = []

            elif self.model_name == "mf":
                # Load encoders
                user_encoder_path = os.path.join(self.context_dir, "mf_user_encoder.pkl")
                item_encoder_path = os.path.join(self.context_dir, "mf_item_encoder.pkl")
                
                with open(user_encoder_path, "rb") as f:
                    self.user_encoder = pickle.load(f)
                    self.logger.info(f"Loaded user encoder from {user_encoder_path}")
                    
                with open(item_encoder_path, "rb") as f:
                    self.item_encoder = pickle.load(f)
                    self.logger.info(f"Loaded item encoder from {item_encoder_path}")

            elif self.model_name == "tfrs":
                # Load candidates dataframe for item lookup
                candidates_path = os.path.join(self.context_dir, "tfrs_candidates.csv")
                if os.path.exists(candidates_path):
                    self.unique_items_df = pd.read_csv(candidates_path)
                    self.logger.info(f"Loaded {len(self.unique_items_df)} candidates from {candidates_path}")
                else:
                    self.logger.warning(f"Candidates file not found at {candidates_path}")
                    self.unique_items_df = None

        except Exception as e:
            self.logger.error(f"Error loading artifacts: {e}")
            raise CustomException(e)

    def recommend(self, user_id, n_rec=10):
        """
        Generates recommendations for a user.
        
        Args:
            user_id: User identifier (string or numeric)
            n_rec: Number of recommendations to return
            
        Returns:
            List of recommended item names
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
            self.logger.error(f"Error in recommend: {e}")
            raise CustomException(e)

    def _recommend_mf(self, user_id, n_rec):
        """Generate recommendations using Matrix Factorization model."""
        try:
            # Convert to string for consistency
            user_id = str(user_id)
            
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
            
        except Exception as e:
            self.logger.error(f"Error in MF recommendation: {e}")
            raise CustomException(e)

    def _recommend_tfrs(self, user_id, n_rec):
        """
        Generate recommendations using TFRS retrieval index.
        This uses the pre-built index for efficient retrieval.
        """
        try:
            # Convert to string for consistency
            user_id = str(user_id)
            
            # Use the pre-built retrieval index
            # The index was built during training and expects string user_id
            _, titles = self.retrieval_index(tf.constant([user_id]))
            
            # Extract top-k recommendations
            recommended_titles = [t.decode("utf-8") for t in titles[0, :n_rec].numpy()]
            
            return recommended_titles
            
        except Exception as e:
            self.logger.error(f"Error in TFRS recommendation: {e}")
            raise CustomException(e)

    def get_similar_items(self, item_name: str, k=10) -> list:
        """
        Finds similar items using Autoencoder embeddings or TFRS item index.
        
        Args:
            item_name: Name of the item to find similar items for
            k: Number of similar items to return
            
        Returns:
            List of similar item names
        """
        try:
            if self.model_name == "autoencoder":
                return self._get_similar_items_autoencoder(item_name, k)
            elif self.model_name == "tfrs":
                return self._get_similar_items_tfrs(item_name, k)
            else:
                self.logger.warning(f"get_similar_items not implemented for {self.model_name}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error in get_similar_items: {e}")
            raise CustomException(e)

    def _get_similar_items_autoencoder(self, item_name: str, k: int) -> list:
        """Find similar items using Autoencoder embeddings."""
        try:
            if self.X is None or self.indices is None:
                raise ValueError("Autoencoder artifacts (X, indices) not loaded.")

            if item_name not in self.indices.index:
                self.logger.warning(f"Item '{item_name}' not found in index.")
                return []

            # Extract encoder part
            encoder = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("bottleneck").output,
            )

            # Compute embeddings for all items
            embeddings = encoder.predict(self.X, verbose=0)

            # Get index of query item
            idx = self.indices[item_name]
            query_embedding = embeddings[idx].reshape(1, -1)

            # Compute cosine similarity
            sim_scores = cosine_similarity(query_embedding, embeddings).flatten()

            # Get top K (excluding self)
            top_indices = sim_scores.argsort()[-(k + 1):][::-1]

            recommendations = []
            for i in top_indices:
                if i != idx and len(recommendations) < k:
                    recommendations.append(self.global_item_names[i])

            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in autoencoder similar items: {e}")
            raise CustomException(e)

    def _get_similar_items_tfrs(self, item_name: str, k: int) -> list:
        """Find similar items using TFRS item-to-item index."""
        try:
            if self.item_index is None:
                self.logger.warning("TFRS item index not loaded.")
                return []
            
            if self.unique_items_df is None:
                raise ValueError("Unique items dataframe not loaded.")
            
            # Find the item in candidates
            item_row = self.unique_items_df[self.unique_items_df['item_name'] == item_name]
            if item_row.empty:
                self.logger.warning(f"Item '{item_name}' not found in candidates.")
                return []
            
            item_text = item_row.iloc[0]['item_text']
            
            # Query the item index
            # The item_index expects dict input with item_name and item_text
            query_input = {
                'item_name': tf.constant([str(item_name)]),
                'item_text': tf.constant([str(item_text)])
            }
            
            _, similar_titles = self.item_index(query_input)
            
            # Extract top-k similar items (excluding self)
            similar_items = []
            for t in similar_titles[0, :k+1].numpy():
                title = t.decode("utf-8")
                if title != item_name and len(similar_items) < k:
                    similar_items.append(title)
            
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Error in TFRS similar items: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    # Example Usage
    try:
        print("=" * 50)
        print("Testing TFRS Pipeline")
        print("=" * 50)
        
        # TFRS Recommendations
        pipeline_tfrs = PredictionPipeline(model_name="tfrs")
        user_recommendations = pipeline_tfrs.recommend(user_id="76561197970982479", n_rec=5)
        print(f"\nTFRS Recommendations for user '76561197970982479':")
        for i, item in enumerate(user_recommendations, 1):
            print(f"{i}. {item}")
        
        # TFRS Similar Items
        if pipeline_tfrs.unique_items_df is not None and len(pipeline_tfrs.unique_items_df) > 0:
            sample_item = pipeline_tfrs.unique_items_df.iloc[0]['item_name']
            similar_items = pipeline_tfrs.get_similar_items(item_name=sample_item, k=5)
            print(f"\nSimilar items to '{sample_item}':")
            for i, item in enumerate(similar_items, 1):
                print(f"{i}. {item}")
        
        print("\n" + "=" * 50)
        print("Testing MF Pipeline")
        print("=" * 50)
        
        # MF Recommendations
        pipeline_mf = PredictionPipeline(model_name='mf')
        mf_recommendations = pipeline_mf.recommend(user_id="76561197970982479", n_rec=5)
        print(f"\nMF Recommendations for user '76561197970982479':")
        for i, item in enumerate(mf_recommendations, 1):
            print(f"{i}. {item}")
        
        print("\n" + "=" * 50)
        print("Testing Autoencoder Pipeline")
        print("=" * 50)
        
        # Autoencoder Similar Items
        pipeline_ae = PredictionPipeline(model_name='autoencoder')
        if len(pipeline_ae.global_item_names) > 0:
            sample_item = pipeline_ae.global_item_names[0]
            ae_similar = pipeline_ae.get_similar_items(item_name=sample_item, k=5)
            print(f"\nAutoencoder similar items to '{sample_item}':")
            for i, item in enumerate(ae_similar, 1):
                print(f"{i}. {item}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise CustomException(e)
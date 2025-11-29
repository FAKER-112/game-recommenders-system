"""
Model Builder Script
--------------------
This script defines the `ModelBuilder` class, which is responsible for constructing and compiling
three different types of recommendation models:

1.  **Autoencoder (Content-Based Filtering)**:
    -   **Data Prep**: Uses TF-IDF to vectorize item descriptions (`item_text`).
    -   **Model**: A symmetric autoencoder that learns compressed representations (embeddings) of items.
    -   **Goal**: Reconstruct item features; useful for finding similar items based on content.

2.  **Matrix Factorization (Collaborative Filtering)**:
    -   **Data Prep**: Encodes User IDs and Item Names into integer indices.
    -   **Model**: Learns separate embedding vectors for Users and Items. Computes the dot product to predict ratings.
    -   **Goal**: Predict user preference (rating) for unseen items.

3.  **TensorFlow Recommenders (TFRS) (Retrieval)**:
    -   **Data Prep**: Converts inputs to strings and creates TensorFlow Datasets. Builds vocabularies for User IDs, Item Names, and Item Text.
    -   **Model**: A Two-Tower architecture (User Tower & Item Tower) that learns to map users and items into a shared embedding space.
    -   **Goal**: Efficiently retrieve top-k relevant items for a user from the entire catalog.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow_recommenders as tfrs
from typing import Tuple, Any, Dict, List


class ModelBuilder:
    """
    Class to build and prepare data for different recommendation models:
    1. Autoencoder (Content-Based)
    2. Matrix Factorization (Collaborative Filtering)
    3. TensorFlow Recommenders (Retrieval)
    """

    def __init__(self):
        pass

    # ==========================================
    # 1. AUTOENCODER (Content-Based)
    # ==========================================

    def prepare_data_autoencoder(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Series,list, int]:
        """
        Prepares data for the Autoencoder model using TF-IDF on item text.
        """
        # Create unique game dataframes for content extraction
        train_content = df.drop_duplicates(subset="item_name").reset_index(drop=True)
        indices = pd.Series(train_content.index, index=train_content['item_name']).drop_duplicates()
        
        # Convert text to numbers (TF-IDF)
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf.fit(train_content["item_text"].fillna(""))
        X = tfidf.transform(train_content["item_text"].fillna("")).toarray()
        
        input_dim = X.shape[1]
        global_item_names = train_content['item_name'].tolist()

        return X, indices, global_item_names, input_dim
    def build_autoencoder_model(self, input_dim: int, encoding_dim: int = 64) -> Model:
        """
        Builds an Autoencoder Neural Network.
        """
        input_layer = layers.Input(shape=(input_dim,))

        # Encoder
        encoded = layers.Dense(512, activation="relu")(input_layer)
        encoded = layers.Dense(256, activation="relu")(encoded)
        encoded = layers.Dense(encoding_dim, activation="relu", name="bottleneck")(
            encoded
        )

        # Decoder
        decoded = layers.Dense(256, activation="relu")(encoded)
        decoded = layers.Dense(512, activation="relu")(decoded)
        output_layer = layers.Dense(input_dim, activation="sigmoid")(decoded)

        # Compile
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

        return autoencoder

    # ==========================================
    # 2. MATRIX FACTORIZATION (Collaborative)
    # ==========================================

    def prepare_data_mf(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Prepares data for Matrix Factorization.
        Encodes user_id and item_name to integers.
        Fits encoders on the UNION of train and test to handle all IDs.
        """
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        # Fit on all unique users/items to ensure consistent encoding
        all_users = pd.concat([train_df["user_id"], test_df["user_id"]]).unique()
        all_items = pd.concat([train_df["item_name"], test_df["item_name"]]).unique()

        user_encoder.fit(all_users)
        item_encoder.fit(all_items)

        # Transform
        train_user_ids = user_encoder.transform(train_df["user_id"])
        train_item_ids = item_encoder.transform(train_df["item_name"])

        test_user_ids = user_encoder.transform(test_df["user_id"])
        test_item_ids = item_encoder.transform(test_df["item_name"])

        num_users = len(user_encoder.classes_)
        num_items = len(item_encoder.classes_)

        return (
            train_user_ids,
            train_item_ids,
            test_user_ids,
            test_item_ids,
            num_users,
            num_items,
            user_encoder,
            item_encoder
        )

    def build_mf_model(
        self, num_users: int, num_items: int, embedding_size: int = 50
    ) -> Model:
        """
        Builds a Matrix Factorization model using Keras Embeddings.
        """
        user_input = layers.Input(shape=(1,), name="user_input")
        game_input = layers.Input(shape=(1,), name="game_input")

        # Embeddings
        user_embedding = layers.Embedding(
            num_users, embedding_size, name="user_embedding"
        )(user_input)
        game_embedding = layers.Embedding(
            num_items, embedding_size, name="game_embedding"
        )(game_input)

        # Flatten
        user_vec = layers.Flatten()(user_embedding)
        game_vec = layers.Flatten()(game_embedding)

        # Dot Product
        dot_product = layers.Dot(axes=1)([user_vec, game_vec])

        # Output
        output = layers.Dense(1, activation="sigmoid")(dot_product)

        model = Model(inputs=[user_input, game_input], outputs=output)
        model.compile(optimizer="adam", loss="mse")

        return model

    # ==========================================
    # 3. TENSORFLOW RECOMMENDERS (Retrieval)
    # ==========================================

    def prepare_data_tfrs(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepares data for TFRS model.
        Returns a dictionary containing datasets and vocabularies.
        """
        # Ensure strings
        for df in [train_df, test_df]:
            df["user_id"] = df["user_id"].astype(str)
            df["item_name"] = df["item_name"].astype(str)
            df["item_text"] = df["item_text"].astype(str)

        # Create Datasets
        def create_interactions(df):
            return tf.data.Dataset.from_tensor_slices(
                {
                    "user_id": df["user_id"].values,
                    "item_name": df["item_name"].values,
                    "item_text": df["item_text"].values,
                    "rating": df["rating"].values.astype(np.float32),
                }
            )

        train_ds = create_interactions(train_df)
        test_ds = create_interactions(test_df)

        # Items Dataset (Catalog) - Use all unique items from both sets
        all_items_df = (
            pd.concat([train_df, test_df])
            .drop_duplicates(subset="item_name")
            .reset_index(drop=True)
        )
        items = tf.data.Dataset.from_tensor_slices(
            {
                "item_name": all_items_df["item_name"].values,
                "item_text": all_items_df["item_text"].values,
            }
        )

        # Vocabularies - Adapt on ALL data to ensure coverage
        all_interactions = train_ds.concatenate(test_ds)

        user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        user_ids_vocabulary.adapt(all_interactions.map(lambda x: x["user_id"]))

        item_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
        item_titles_vocabulary.adapt(items.map(lambda x: x["item_name"]))

        # Text Vectorization
        text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000)
        text_vectorizer.adapt(items.map(lambda x: x["item_text"]))
        all_df = pd.concat([train_df, test_df])
        unique_items_df = all_df.drop_duplicates(subset=['item_name'])[['item_name', 'item_text']]
        return {
            "train_ds": train_ds,
            "test_ds": test_ds,
            "items": items,
            "user_ids_vocabulary": user_ids_vocabulary,
            "item_titles_vocabulary": item_titles_vocabulary,
            "text_vectorizer": text_vectorizer,
            "unique_items_df": unique_items_df
        }

    def build_tfrs_model(self, data_dict: Dict[str, Any]) -> tfrs.Model:
        """
        Builds the TFRS Retrieval Model.
        """
        user_ids_vocabulary = data_dict["user_ids_vocabulary"]
        item_titles_vocabulary = data_dict["item_titles_vocabulary"]
        text_vectorizer = data_dict["text_vectorizer"]
        items = data_dict["items"]

        # Define User Tower
        class UserModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.user_embedding = tf.keras.Sequential(
                    [
                        user_ids_vocabulary,
                        tf.keras.layers.Embedding(
                            user_ids_vocabulary.vocabulary_size(), 64
                        ),
                    ]
                )

            def call(self, inputs):
                return self.user_embedding(inputs)

        # Define Item Tower
        class ItemModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.title_embedding = tf.keras.Sequential(
                    [
                        item_titles_vocabulary,
                        tf.keras.layers.Embedding(
                            item_titles_vocabulary.vocabulary_size(), 32
                        ),
                    ]
                )
                self.text_embedding = tf.keras.Sequential(
                    [
                        text_vectorizer,
                        tf.keras.layers.Embedding(10000, 32, mask_zero=True),
                        tf.keras.layers.GlobalAveragePooling1D(),
                    ]
                )
                self.dense = tf.keras.Sequential(
                    [tf.keras.layers.Dense(64, activation="relu")]
                )

            def call(self, titles, texts):
                title_emb = self.title_embedding(titles)
                text_emb = self.text_embedding(texts)
                concat = tf.concat([title_emb, text_emb], axis=1)
                return self.dense(concat)

        # Define Main Model
        class SteamTFRSModel(tfrs.Model):
            def __init__(self):
                super().__init__()
                self.user_model = UserModel()
                self.item_model = ItemModel()

                # Candidate Embeddings for Retrieval
                self.task = tfrs.tasks.Retrieval(
                    metrics=tfrs.metrics.FactorizedTopK(
                        candidates=items.batch(128).map(self.item_model)
                    )
                )

            def compute_loss(self, features, training=False):
                user_emb = self.user_model(features["user_id"])
                item_emb = self.item_model(features["item_name"], features["item_text"])
                return self.task(user_emb, item_emb)

        return SteamTFRSModel()

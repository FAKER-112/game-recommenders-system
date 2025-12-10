"""
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
    ) -> Tuple[np.ndarray, pd.Series, list, int]:
        """
        Prepares data for the Autoencoder model.

        Input:
            - df (pd.DataFrame): The transformed dataset containing 'item_text' and 'item_name'.

        Output:
            - X (np.ndarray): The TF-IDF feature matrix (samples x max_features).
            - indices (pd.Series): Mapping from item names to dataframe indices.
            - global_item_names (list): List of unique item names corresponding to the rows of X.
            - input_dim (int): The number of features in the TF-IDF vectors.

        Logic:
            1.  **Deduplication**: Drops duplicate items to ensure unique content profiles.
            2.  **Vectorization**: Uses TfidfVectorizer (max 5000 features) to convert
                textual descriptions (`item_text`) into numerical vectors.
            3.  **Metadata Extraction**: Extracts indices and names for later retrieval.
        """
        # Create unique game dataframes for content extraction
        train_content = df.drop_duplicates(subset="item_name").reset_index(drop=True)
        indices = pd.Series(
            train_content.index, index=train_content["item_name"]
        ).drop_duplicates()

        # Convert text to numbers (TF-IDF)
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf.fit(train_content["item_text"].fillna(""))
        X = tfidf.transform(train_content["item_text"].fillna("")).toarray()

        input_dim = X.shape[1]
        global_item_names = train_content["item_name"].tolist()

        return X, indices, global_item_names, input_dim

    def build_autoencoder_model(self, input_dim: int, encoding_dim: int = 64) -> Model:
        """
        Builds the Autoencoder Neural Network architecture.

        Input:
            - input_dim (int): Dimensionality of the input textual vectors (from TF-IDF).
            - encoding_dim (int): Size of the compressed latent representation (bottleneck).

        Output:
            - model (Model): A compiled Keras model.

        Logic:
            1.  **Encoder**: Compresses input from `input_dim` -> 512 -> 256 -> `encoding_dim`.
                Uses ReLU activation.
            2.  **Decoder**: Reconstructs input from `encoding_dim` -> 256 -> 512 -> `input_dim`.
                Uses Sigmoid activation at the output (assuming TF-IDF is normalized/bounded).
            3.  **Compilation**: Uses Adam optimizer and Binary Crossentropy loss (suitable for
                reconstruction tasks where inputs are essentially probability-like or normalized).
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
        Prepares data for Matrix Factorization by integer encoding users and items.

        Input:
            - train_df (pd.DataFrame): Training data with 'user_id' and 'item_name'.
            - test_df (pd.DataFrame): Testing data with 'user_id' and 'item_name'.

        Output:
            - train/test_user_ids (np.ndarray): Encoded integer IDs for users.
            - train/test_item_ids (np.ndarray): Encoded integer IDs for items.
            - num_users/num_items (int): Total count of unique users and items.
            - user/item_encoder (LabelEncoder): Fitted encoders for inverse transformation.

        Logic:
            1.  **Union of IDs**: Concatenates train and test sets to find ALL unique users and items.
                This ensures that the encoders handle the full vocabulary and don't fail on unseen IDs in test.
            2.  **Encoding**: Fits LabelEncoder to map strings -> integers (0 to N-1).
            3.  **Transformation**: Converts the columns in both train and test dataframes to integers.
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
            item_encoder,
        )

    def build_mf_model(
        self, num_users: int, num_items: int, embedding_size: int = 50
    ) -> Model:
        """
        Builds a standard Matrix Factorization model.

        Input:
            - num_users (int): Total number of unique users.
            - num_items (int): Total number of unique items.
            - embedding_size (int): Dimension of the embedding vectors.

        Output:
            - model (Model): A compiled Keras model.

        Logic:
            1.  **Inputs**: Accepts two integer inputs: user ID and item ID.
            2.  **Embeddings**: Looks up learnable vectors (size `embedding_size`) for the user and item.
            3.  **Interaction**: Computes the Dot Product of the user and item vectors.
                This captures the similarity/affinity between user and item.
            4.  **Regression**: Outputs a single scalar (predicted rating).
            5.  **Compilation**: Uses Adam optimizer and Mean Squared Error (MSE) loss.
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
        output = layers.Dense(1)(dot_product)

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
        Prepares data for TensorFlow Recommenders (Two-Tower Retrieval).

        Input:
            - train/test_df (pd.DataFrame): Dataframes containing interactions.

        Output:
            - data_dict (dict): Contains 'train_ds', 'test_ds' (tf.data.Dataset),
              'items' (tf.data.Dataset of catalog), and fitted vocabularies/vectorizers.

        Logic:
            1.  **Type Casting**: Ensures all IDs and text are distinct strings.
            2.  **Dataset Creation**: Converts Pandas DataFrames into `tf.data.Dataset` objects.
                - Interaction datasets contain {user_id, item_name, item_text, rating}.
                - Items dataset (catalog) contains unique {item_name, item_text}.
            3.  **Vocabulary Adaptation**:
                - `StringLookup` for User IDs and Item Names: Learn the mapping from string -> int ID.
                - `TextVectorization` for Item Text: Learns tokens from descriptions for content embedding.
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
        unique_items_df = all_df.drop_duplicates(subset=["item_name"])[
            ["item_name", "item_text"]
        ]
        return {
            "train_ds": train_ds,
            "test_ds": test_ds,
            "items": items,
            "user_ids_vocabulary": user_ids_vocabulary,
            "item_titles_vocabulary": item_titles_vocabulary,
            "text_vectorizer": text_vectorizer,
            "unique_items_df": unique_items_df,
        }

    def build_tfrs_model(self, data_dict: Dict[str, Any]) -> tfrs.Model:
        """
        Builds the TFRS Two-Tower Retrieval Model.

        Input:
            - data_dict (dict): The dictionary produced by `prepare_data_tfrs`, containing
              datasets, vocabularies, and vectorizers.

        Output:
            - model (tfrs.Model): An instance of the custom `SteamTFRSModel`.

        Logic:
            1.  **User Tower**:
                - Input: User ID.
                - Layers: StringLookup -> Embedding (dim=64).
                - Output: User representation vector.
            2.  **Item Tower**:
                - Input: Item Name and Item Text.
                - Logic:
                    - Embeds Name (StringLookup -> Embedding).
                    - Embeds Text (TextVectorization -> Embedding -> GlobalAveragePooling).
                    - Concatenates Name + Text embeddings -> Dense layer (dim=64).
                - Output: Item representation vector.
            3.  **Task (Loss)**: `tfrs.tasks.Retrieval`.
                - Uses `FactorizedTopK` metric to check if the true item is in the top-K retrieved items.
                - Computes loss based on the similarity (dot product) between User Tower output
                  and Item Tower output.
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

            def call(self, inputs):
                # Handle dict input from .map() on items dataset
                if isinstance(inputs, dict):
                    titles = inputs["item_name"]
                    texts = inputs["item_text"]
                else:
                    # Fallback for direct calls
                    titles = inputs
                    texts = None
                title_emb = self.title_embedding(titles)
                if texts is not None:
                    text_emb = self.text_embedding(texts)
                    concat = tf.concat([title_emb, text_emb], axis=1)
                else:
                    concat = title_emb
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
                item_emb = self.item_model(
                    {
                        "item_name": features["item_name"],
                        "item_text": features["item_text"],
                    }
                )
                return self.task(user_emb, item_emb)

        return SteamTFRSModel()

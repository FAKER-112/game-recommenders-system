"""
Feature Engineering Module

This script defines the FeatureEngineeringService class, which transforms cleaned data
into a format suitable for machine learning models. It handles feature creation,
text processing, and dataset splitting.

Logic of Operation:
1.  **Initialization**:
    - Loads configuration to locate the cleaned data file and define output paths
      for transformed data (train, test, and full dataset).
2.  **Feature Engineering (in `run` method)**:
    - **Data Loading**: Reads the cleaned CSV file.
    - **Rating Creation**: Transforms the 'playtime' feature using `log(1+x)` to
      create a 'rating' implicit feedback signal, reducing skewness.
    - **Text Processing**:
        - Parses 'genres' and 'tags' columns (handling strings, lists, and malformed data).
        - Joins them into comma-separated strings for storage.
        - Creates a unified `item_text` column by combining genres and tags,
          converting to lowercase, and replacing commas with spaces. This is useful
          for creating content vectors (e.g., using TF-IDF or embeddings).
    - **Data Cleaning**: Drops intermediate columns and rows with empty text information.
3.  **Data Splitting & Saving**:
    - Saves the full transformed dataset.
    - Splits the data into training (80%) and testing (20%) sets.
    - Saves the train and test sets to the configured paths.
"""

import pandas as pd
import numpy as np
import os
import ast
from pathlib import Path
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.utils import load_config
from sklearn.model_selection import train_test_split


class FeatureEngineeringService:
    """
    Service class for performing feature engineering on the cleaned dataset.
    """

    def __init__(self, raw_data_yaml: str = str(Path("configs/config.yaml"))):
        """
        Initialize the FeatureEngineeringService.

        Args:
            raw_data_yaml (str): Path to the configuration YAML file.
        """
        try:
            self.config = load_config(raw_data_yaml)
            self.logger = logger

            # Load feature engineering config
            feat_cfg = self.config.get("feature_engineering", {})
            self.root_dir = feat_cfg.get("root_dir")
            self.cleaned_data_path = feat_cfg.get("cleaned_data_path")
            self.transformed_train_path = feat_cfg.get("transformed_train_path")
            self.transformed_test_path = feat_cfg.get("transformed_test_path")
            self.transformed_data_path = feat_cfg.get("transformed_data_path")

        except Exception as e:
            raise CustomException(e)

    def run(self):
        """
        Execute the feature engineering pipeline.

        Steps:
        1. Load cleaned data.
        2. Create 'rating' feature from 'playtime'.
        3. Process 'genres' and 'tags' into string format.
        4. Create 'item_text' for content-based filtering.
        5. Split data into train and test sets.
        6. Save transformed datasets.
        """
        try:
            self.logger.info("Starting feature engineering process...")

            # Load data
            if not os.path.exists(self.cleaned_data_path):
                raise FileNotFoundError(
                    f"Cleaned data file not found: {self.cleaned_data_path}"
                )

            df = pd.read_csv(self.cleaned_data_path)
            self.logger.info(f"Loaded data with shape: {df.shape}")

            # Feature Creation: Rating
            # Log transformation of playtime to reduce skewness
            df["playtime"] = df["playtime"].fillna(0).astype(int)
            df["rating"] = np.log1p(df["playtime"])  # log(1+x) is safer and cleaner

            def fix_list(x):
                """
                Helper to ensure list columns are correctly parsed.
                Handles strings, lists, and malformed data.
                """
                if isinstance(x, list):
                    return x
                if pd.isna(x):
                    return []
                if isinstance(x, str):
                    x = x.replace("&amp;", "&")  # fix HTML escapes
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError):
                        return [x]  # fallback: treat as single item list
                return [str(x)]

            # Process genres
            df["genres_1"] = (
                df["genres"]
                .apply(fix_list)
                .apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
            )

            # Process tags
            df["tags_1"] = (
                df["tags"]
                .apply(fix_list)
                .apply(lambda lst: ", ".join(lst) if isinstance(lst, list) else "")
            )

            # Clean up columns
            df.drop(columns=["genres", "tags"], inplace=True)
            df.rename(columns={"genres_1": "genres", "tags_1": "tags"}, inplace=True)

            if "title" in df.columns:
                df.drop(columns="title", inplace=True)

            # Create item_text for NLP/Similarity tasks
            # Combine genres and tags, replace commas with spaces, convert to lower case
            df["item_text"] = (
                (
                    df["genres"].fillna("").str.replace(",", " ")
                    + " "
                    + df["tags"].fillna("").str.replace(",", " ")
                )
                .str.lower()
                .str.strip()
            )

            # Filter out rows with no text info
            df = df[df["item_text"] != ""]

            # Save full transformed data
            df.to_csv(self.transformed_data_path, index=False)
            self.logger.info(f"Transformed data saved to {self.transformed_data_path}")

            # Train/Test Split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.transformed_train_path, index=False)
            test_df.to_csv(self.transformed_test_path, index=False)

            self.logger.info(f"Train data saved to {self.transformed_train_path}")
            self.logger.info(f"Test data saved to {self.transformed_test_path}")
            self.logger.info("Feature engineering completed successfully.")

        except Exception as e:
            raise CustomException(e)

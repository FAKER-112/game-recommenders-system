"""
Data Cleaning and Merging Module

This script defines the CleanDataService class, which is responsible for processing
raw data files into a consolidated, clean dataset ready for analysis or modeling.

Logic of Operation:
1.  **Initialization**:
    - Loads configuration to locate raw data files (User Items and Steam Games)
      and define the output directory for processed data.
    - Validates the existence of required raw files.

2.  **Data Processing (in `run` method)**:
    - Checks if the processed output file already exists. If so, skips processing
      to save time.
    - **User Items Processing**:
        - Reads the gzipped user data line-by-line.
        - Parses Python-literal formatted lines to flatten the nested structure
          (one row per user-item interaction).
        - Extracts `user_id`, `item_id`, `playtime`, and `item_name`.
    - **Steam Games Processing**:
        - Reads and parses the gzipped games data.
        - Filters for relevant metadata: `id`, `genres`, `tags`, and `title`.
        - Standardizes column names (renames `id` to `item_id`).
    - **Merging**:
        - Merges the user-item interactions with game metadata on `item_id`
          using a left join.
    - **Output**:
        - Saves the final merged dataset to a CSV file in the processed directory
          (e.g., `data/processed/australian_users_items_merged.csv`).
"""

import os
import sys
import gzip
import ast
import pandas as pd
from pathlib import Path
from src.utils.utils import load_config
from src.utils.exception import CustomException
from src.utils.logger import logger


class CleanDataService:
    def __init__(self, raw_data_yaml: str = str(Path("configs/config.yaml"))):
        try:
            self.config = load_config(raw_data_yaml)
            self.logger = logger

            # Load configurations
            ingest_cfg = self.config.get("data_ingestion", {})
            clean_cfg = self.config.get("data_cleaning", {})

            self.raw_data_dir = clean_cfg.get("raw_data_dir", "data/raw")
            self.processed_dir = clean_cfg.get("root_dir", "data/processed")

            # Get filenames from ingestion config URLs to ensure consistency
            self.user_items_filename = ingest_cfg.get(
                "user_item_dataset_download_url", ""
            ).split("/")[-1]
            self.steam_games_filename = ingest_cfg.get(
                "steam_game_dataset_download_url", ""
            ).split("/")[-1]

            # Validate raw files exist
            self.user_items_path = os.path.join(
                self.raw_data_dir, self.user_items_filename
            )
            self.steam_games_path = os.path.join(
                self.raw_data_dir, self.steam_games_filename
            )
            print("User items filename:", self.user_items_filename)
            print("Steam games filename:", self.steam_games_filename)

            if not os.path.exists(self.user_items_path):
                raise FileNotFoundError(f"File {self.user_items_path} does not exist")
            if not os.path.exists(self.steam_games_path):
                raise FileNotFoundError(f"File {self.steam_games_path} does not exist")

            # Ensure processed directory exists
            os.makedirs(self.processed_dir, exist_ok=True)

        except Exception as e:
            raise CustomException(e)

    def run(self):
        try:
            self.logger.info("Starting data cleaning process...")

            # 1. Process User Items
            self.logger.info(f"Processing {self.user_items_filename}...")
            rows = []
            output_path = os.path.join(
                self.processed_dir, "australian_users_items_merged.csv"
            )
            if os.path.exists(output_path):
                self.logger.info("data has been cleaned")
            else:

                with gzip.open(self.user_items_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        try:
                            user_data = ast.literal_eval(line.strip())
                            user_id = user_data["user_id"]
                            for item in user_data["items"]:
                                rows.append(
                                    {
                                        "user_id": user_id,
                                        "item_id": item["item_id"],
                                        "playtime": item["playtime_forever"],
                                        "item_name": item["item_name"],
                                    }
                                )
                        except Exception as e:
                            self.logger.warning(
                                f"Error parsing line in user items: {e}"
                            )
                            continue

                df_users = pd.DataFrame(rows)
                self.logger.info(f"User items loaded. Shape: {df_users.shape}")

                # 2. Process Steam Games
                self.logger.info(f"Processing {self.steam_games_filename}...")
                steam_data = []
                with gzip.open(self.steam_games_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        try:
                            game_dict = ast.literal_eval(line)
                            steam_data.append(game_dict)
                        except (ValueError, SyntaxError):
                            continue

                steam_df = pd.DataFrame(steam_data)

                # Keep only relevant columns
                cols_to_keep = ["id", "genres", "tags", "title"]
                # Filter existing columns only
                existing_cols = [c for c in cols_to_keep if c in steam_df.columns]
                steam_df = steam_df[existing_cols]

                # Rename 'id' to 'item_id'
                if "id" in steam_df.columns:
                    steam_df = steam_df.rename(columns={"id": "item_id"})

                steam_df["item_id"] = steam_df["item_id"].astype(str)
                self.logger.info(f"Steam games loaded. Shape: {steam_df.shape}")

                # 3. Merge Data
                self.logger.info("Merging datasets...")
                # Ensure item_id in df_users is string for merging
                df_users["item_id"] = df_users["item_id"].astype(str)

                df_merged = df_users.merge(steam_df, on="item_id", how="left")

                output_path = os.path.join(
                    self.processed_dir, "australian_users_items_merged.csv"
                )
                df_merged.to_csv(output_path, index=False)

                self.logger.info(f"Data cleaning completed. Saved to {output_path}")

        except Exception as e:
            raise CustomException(e)

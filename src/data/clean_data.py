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
                        self.logger.warning(f"Error parsing line in user items: {e}")
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

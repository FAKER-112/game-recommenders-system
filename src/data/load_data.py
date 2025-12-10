"""
Data Ingestion Module

This script defines the LoadDataService class, which facilitates the initial data ingestion phase
for the project. Its primary purpose is to ensure that the necessary raw datasets are available
locally for downstream processing.

Logic of Operation:
1.  **Configuration Loading**: The class initializes by loading a YAML configuration file
    (defaulting to `configs/config.yaml`) to retrieve data ingestion settings, including
    download URLs and the target directory for raw data.
2.  **Environment Setup**: It verifies the existence of the configured raw data directory
    and creates it if it does not exist.
3.  **Data Retrieval**: In the `run` method, the service iterates through the list of target URLs:
    - It extracts the filename from the URL.
    - It checks if the file already exists in the local raw data directory.
    - If the file is missing, it downloads it using the `download_file` utility.
    - If the file exists, it skips the download to prevent redundancy.
4.  **Logging & Error Handling**: The process provides logging for each step (check/download)
    and wraps execution in a try-catch block to raise `CustomException` on failure.
"""

import os
import sys
from pathlib import Path
from src.utils.utils import download_file, load_config
from src.utils.exception import CustomException
from src.utils.logger import logger


class LoadDataService:
    def __init__(self, raw_data_yaml: str = str(Path("configs/config.yaml"))):
        try:
            self.config = load_config(raw_data_yaml)
            self.logger = logger

            load_cfg = self.config.get("data_ingestion", {})

            # Store URLs in a list for easier iteration
            self.urls = [
                load_cfg.get(
                    "user_item_dataset_download_url",
                    "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz",
                ),
                load_cfg.get(
                    "steam_game_dataset_download_url",
                    "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
                ),
            ]
            self.raw_data_dir = load_cfg.get("raw_data_dir", "data/raw")

            if self.raw_data_dir:
                os.makedirs(self.raw_data_dir, exist_ok=True)
            else:
                raise ValueError("raw_data_dir not found in configuration")

        except Exception as e:
            # CustomException only takes the error object
            raise CustomException(e)

    def run(self):
        try:
            self.logger.info("Starting data ingestion process...")

            for url in self.urls:
                # if not url:
                #     self.logger.warning("Encountered empty URL in configuration. Skipping.")
                #     continue
                self.logger.info(f"url={url}")
                filename = url.split("/")[-1]
                file_path = os.path.join(self.raw_data_dir, filename)

                if os.path.exists(file_path):
                    self.logger.info(f"File already exists: {file_path}")
                else:
                    self.logger.info(
                        f"File does not exist: {file_path}. Downloading..."
                    )
                    download_file(url, self.raw_data_dir)
                    self.logger.info(f"Successfully downloaded: {filename}")

            self.logger.info("Data ingestion completed.")

        except Exception as e:
            raise CustomException(e)

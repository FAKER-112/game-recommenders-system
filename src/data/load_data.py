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
                load_cfg.get("user_item_dataset_download_url"),
                load_cfg.get("steam_game_dataset_download_url")
            ]
            self.raw_data_dir = load_cfg.get("raw_data_dir")

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
                if not url:
                    self.logger.warning("Encountered empty URL in configuration. Skipping.")
                    continue

                filename = url.split('/')[-1]
                file_path = os.path.join(self.raw_data_dir, filename)

                if os.path.exists(file_path):
                    self.logger.info(f"File already exists: {file_path}")
                else:
                    self.logger.info(f"File does not exist: {file_path}. Downloading...")
                    download_file(url, self.raw_data_dir)
                    self.logger.info(f"Successfully downloaded: {filename}")

            self.logger.info("Data ingestion completed.")
            
        except Exception as e:
            raise CustomException(e)
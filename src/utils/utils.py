import requests
import os
import sys
import pandas as pd 
import numpy as np
import yaml
from src.utils.logger import logger
from pathlib import Path
from src.utils.exception import CustomException

def download_file(url, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        logger.info(f"Created directory: {dest_folder}")

    # Get the file name from the URL
    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)

    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes (e.g., 404)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f"Downloaded {filename} to {file_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
        raise  # Re-raise the exception for the caller to handle if needed

    return file_path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise CustomException(FileNotFoundError(f"Config file not found at {config_path}"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
import requests
import os
import sys
import pandas as pd 
import numpy as np
import yaml
import math
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
        raise CustomException(e) # Re-raise the exception for the caller to handle if needed

    return file_path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise CustomException(FileNotFoundError(f"Config file not found at {config_path}"))
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config




# --- 1. METRIC FUNCTIONS ---


def calculate_precision_at_k(recommended_items, ground_truth_items, k=10):
    """Calculates Precision@K."""
    recommended_k = recommended_items[:k]
    relevant_count = len(set(recommended_k) & set(ground_truth_items))
    return relevant_count / k


def calculate_ap_at_k(recommended_items, ground_truth_items, k=10):
    """Calculates Average Precision@K."""
    if not ground_truth_items:
        return 0.0

    relevant_count = 0
    running_sum = 0.0

    for i, item in enumerate(recommended_items[:k]):
        if item in ground_truth_items:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            running_sum += precision_at_i

    if relevant_count == 0:
        return 0.0
    return running_sum / min(len(ground_truth_items), k)


def calculate_ndcg_at_k(recommended_items, ground_truth_items, k=10):
    """Calculates NDCG@K."""
    dcg = 0.0
    idcg = 0.0

    # 1. Calculate DCG
    for i, item in enumerate(recommended_items[:k]):
        if item in ground_truth_items:
            dcg += 1.0 / math.log2(i + 2)

    # 2. Calculate IDCG
    num_relevant = min(len(ground_truth_items), k)
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg

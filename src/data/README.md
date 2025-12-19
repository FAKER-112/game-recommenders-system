# Data Pipeline

This directory contains the data processing pipeline for the game recommendation system. The pipeline consists of three sequential modules that transform raw data into machine learning-ready features.

## Overview

The data pipeline performs the following operations:
1. **Data Ingestion** - Downloads and stores raw datasets
2. **Data Cleaning** - Processes and merges raw data files
3. **Feature Engineering** - Creates features and prepares train/test splits

## Modules

### 1. load_data.py

**Purpose**: Initial data ingestion and download management.

**Class**: `LoadDataService`

**Functionality**:
- Loads configuration from `configs/config.yaml` to retrieve data source URLs and target directories
- Creates the raw data directory structure if it doesn't exist
- Downloads missing datasets from configured URLs
- Skips downloads for files that already exist locally
- Provides comprehensive logging and error handling via `CustomException`

**Output**: Raw data files stored in the configured raw data directory

---

### 2. clean_data.py

**Purpose**: Data cleaning, processing, and merging.

**Class**: `CleanDataService`

**Functionality**:
- **User Items Processing**:
  - Reads gzipped user-item interaction data
  - Parses and flattens nested Python-literal formatted data
  - Extracts key fields: `user_id`, `item_id`, `playtime`, `item_name`
  
- **Steam Games Processing**:
  - Reads gzipped game metadata
  - Filters relevant features: `id`, `genres`, `tags`, `title`
  - Standardizes column names (`id` → `item_id`)

- **Data Merging**:
  - Performs left join of user interactions with game metadata on `item_id`
  - Creates a consolidated dataset combining user behavior and game attributes

**Output**: Merged CSV file (e.g., `data/processed/australian_users_items_merged.csv`)

---

### 3. feature_engineering.py

**Purpose**: Feature transformation and dataset preparation for machine learning.

**Class**: `FeatureEngineeringService`

**Functionality**:
- **Rating Creation**: 
  - Applies log transformation `log(1+x)` to `playtime` to create implicit `rating` signal
  - Reduces skewness in the target variable

- **Text Processing**:
  - Parses `genres` and `tags` columns (handles various formats)
  - Creates `item_text` field by combining genres and tags
  - Normalizes text to lowercase and formats for vectorization (TF-IDF/embeddings)

- **Data Cleaning**:
  - Removes rows with missing text information
  - Drops intermediate processing columns

- **Train/Test Split**:
  - Splits data into training (80%) and testing (20%) sets
  - Saves full transformed dataset and separate train/test files

**Output**: 
- Full transformed dataset
- Training dataset (80%)
- Testing dataset (20%)

## Usage

The modules are designed to run sequentially as part of the data pipeline:

```python
# 1. Load raw data
from load_data import LoadDataService
loader = LoadDataService()
loader.run()

# 2. Clean and merge data
from clean_data import CleanDataService
cleaner = CleanDataService()
cleaner.run()

# 3. Engineer features and create splits
from feature_engineering import FeatureEngineeringService
engineer = FeatureEngineeringService()
engineer.run()
```

## Configuration

All modules rely on `configs/config.yaml` for:
- Data source URLs
- Directory paths (raw, processed, transformed)
- File naming conventions
- Processing parameters

## Error Handling

All modules implement robust error handling:
- Custom exception classes for better debugging
- Comprehensive logging at each processing step
- Validation checks for file existence and data integrity

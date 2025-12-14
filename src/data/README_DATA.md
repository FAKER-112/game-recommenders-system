data loading, processing and feature engineering


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

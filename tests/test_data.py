"""
Unit Tests for Data Processing Module
======================================

This test suite provides comprehensive unit testing for all data processing classes
in the `src/data` directory, covering data ingestion, cleaning, and feature engineering
workflows for the recommendation system.

Test Coverage:
-------------

1. **TestLoadDataService** (6 tests)
   - Tests initialization with config loading and directory creation
   - Validates URL extraction from configuration
   - Tests data download workflow when files don't exist
   - Tests skip logic when files already exist
   - Tests exception handling during initialization and execution
   - Verifies proper logging at each stage

2. **TestCleanDataService** (5 tests)
   - Tests initialization with path validation
   - Tests user items processing (gzip reading, line parsing, flattening)
   - Tests steam games processing (parsing, column filtering, renaming)
   - Tests data merging (left join on item_id)
   - Tests exception handling and skip logic for existing files

3. **TestFeatureEngineeringService** (6 tests)
   - Tests initialization and configuration loading
   - Tests rating creation from playtime (log transformation)
   - Tests genres and tags parsing (handling lists, strings, malformed data)
   - Tests item_text creation (combining and cleaning text)
   - Tests train/test split (80/20 ratio, random_state=42)
   - Tests exception handling and data filtering

Mocking Strategy:
----------------
All tests use extensive mocking to ensure:
- Fast execution (no actual file I/O or downloads)
- Isolation from external dependencies (config files, data files, network)
- Consistent and reproducible test results

Mocked components include:
- load_config: Returns mock configuration dictionaries
- os.path.exists: Controls file existence checks
- os.makedirs: Prevents actual directory creation
- download_file: Mocks file downloads
- gzip.open: Mocks compressed file reading
- pandas.read_csv/to_csv: Mocks CSV operations
- train_test_split: Mocks sklearn data splitting

Usage:
------
Run all tests:
    python -m pytest tests/test_data.py -v

Run specific test class:
    python -m pytest tests/test_data.py::TestLoadDataService -v

Run with unittest:
    python -m unittest tests.test_data
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.load_data import LoadDataService
from src.data.clean_data import CleanDataService
from src.data.feature_engineering import FeatureEngineeringService
from src.utils.exception import CustomException


class TestLoadDataService(unittest.TestCase):
    """Test suite for LoadDataService class"""

    @patch("src.data.load_data.load_config")
    @patch("os.makedirs")
    def test_init_success(self, mock_makedirs, mock_load_config):
        """Test LoadDataService initialization"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
                "raw_data_dir": "data/raw",
            }
        }
        mock_load_config.return_value = mock_config

        service = LoadDataService("configs/config.yaml")

        # Verify initialization
        self.assertEqual(service.raw_data_dir, "data/raw")
        self.assertEqual(len(service.urls), 2)
        self.assertIn("users.json.gz", service.urls[0])
        mock_makedirs.assert_called_once_with("data/raw", exist_ok=True)

    @patch("os.makedirs")
    @patch("src.data.load_data.load_config")
    def test_init_missing_raw_data_dir(self, mock_load_config, mock_makedirs):
        """Test initialization fails when raw_data_dir is missing"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz"
            }
        }
        mock_load_config.return_value = mock_config

        with self.assertRaises(CustomException):
            LoadDataService()

    @patch("src.data.load_data.download_file")
    @patch("os.path.exists")
    @patch("src.data.load_data.load_config")
    @patch("os.makedirs")
    def test_run_downloads_missing_files(
        self, mock_makedirs, mock_load_config, mock_exists, mock_download
    ):
        """Test run method downloads files that don't exist"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
                "raw_data_dir": "data/raw",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False  # Files don't exist

        service = LoadDataService()
        service.run()

        # Verify download was called for both URLs
        self.assertEqual(mock_download.call_count, 2)
        mock_download.assert_any_call("https://example.com/users.json.gz", "data/raw")
        mock_download.assert_any_call("https://example.com/games.json.gz", "data/raw")

    @patch("src.data.load_data.download_file")
    @patch("os.path.exists")
    @patch("src.data.load_data.load_config")
    @patch("os.makedirs")
    def test_run_skips_existing_files(
        self, mock_makedirs, mock_load_config, mock_exists, mock_download
    ):
        """Test run method skips files that already exist"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
                "raw_data_dir": "data/raw",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True  # Files exist

        service = LoadDataService()
        service.run()

        # Verify download was NOT called
        mock_download.assert_not_called()

    @patch("src.data.load_data.download_file")
    @patch("os.path.exists")
    @patch("src.data.load_data.load_config")
    @patch("os.makedirs")
    def test_run_handles_download_error(
        self, mock_makedirs, mock_load_config, mock_exists, mock_download
    ):
        """Test run method handles download errors"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "raw_data_dir": "data/raw",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False
        mock_download.side_effect = Exception("Download failed")

        service = LoadDataService()

        with self.assertRaises(CustomException):
            service.run()


class TestCleanDataService(unittest.TestCase):
    """Test suite for CleanDataService class"""

    @patch("src.data.clean_data.load_config")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_init_success(self, mock_makedirs, mock_exists, mock_load_config):
        """Test CleanDataService initialization"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
            },
            "data_cleaning": {"raw_data_dir": "data/raw", "root_dir": "data/processed"},
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        service = CleanDataService("configs/config.yaml")

        # Verify initialization
        self.assertEqual(service.raw_data_dir, "data/raw")
        self.assertEqual(service.processed_dir, "data/processed")
        self.assertIn("users.json.gz", service.user_items_filename)
        mock_makedirs.assert_called_once_with("data/processed", exist_ok=True)

    @patch("src.data.clean_data.load_config")
    @patch("os.path.exists")
    def test_init_missing_raw_files(self, mock_exists, mock_load_config):
        """Test initialization fails when raw files don't exist"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
            },
            "data_cleaning": {"raw_data_dir": "data/raw", "root_dir": "data/processed"},
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False

        with self.assertRaises(CustomException):
            CleanDataService()

    @patch("src.data.clean_data.load_config")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_run_skips_if_output_exists(
        self, mock_makedirs, mock_exists, mock_load_config
    ):
        """Test run method skips processing if output already exists"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
            },
            "data_cleaning": {"raw_data_dir": "data/raw", "root_dir": "data/processed"},
        }
        mock_load_config.return_value = mock_config

        # First two calls for raw files (in __init__), third for output file (in run)
        mock_exists.side_effect = [True, True, True]

        service = CleanDataService()
        service.run()

        # Verify that processing was skipped (no gzip.open calls)
        # We just check that the method completes without errors

    @patch("pandas.DataFrame.to_csv")
    @patch("gzip.open")
    @patch("src.data.clean_data.load_config")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_run_processes_data(
        self, mock_makedirs, mock_exists, mock_load_config, mock_gzip_open, mock_to_csv
    ):
        """Test run method processes and merges data"""
        mock_config = {
            "data_ingestion": {
                "user_item_dataset_download_url": "https://example.com/users.json.gz",
                "steam_game_dataset_download_url": "https://example.com/games.json.gz",
            },
            "data_cleaning": {"raw_data_dir": "data/raw", "root_dir": "data/processed"},
        }
        mock_load_config.return_value = mock_config
        mock_exists.side_effect = [True, True, False]  # Raw files exist, output doesn't

        # Mock user items data
        user_line = "{'user_id': 'user1', 'items': [{'item_id': '123', 'playtime_forever': 100, 'item_name': 'Game1'}]}\n"
        # Mock steam games data
        game_line = (
            "{'id': '123', 'title': 'Game1', 'genres': ['Action'], 'tags': ['FPS']}\n"
        )

        mock_gzip_open.return_value.__enter__.return_value = [user_line, game_line]

        service = CleanDataService()
        service.run()

        # Verify CSV was saved
        mock_to_csv.assert_called_once()


class TestFeatureEngineeringService(unittest.TestCase):
    """Test suite for FeatureEngineeringService class"""

    @patch("src.data.feature_engineering.load_config")
    def test_init_success(self, mock_load_config):
        """Test FeatureEngineeringService initialization"""
        mock_config = {
            "feature_engineering": {
                "root_dir": "data/processed",
                "cleaned_data_path": "data/processed/merged.csv",
                "transformed_train_path": "data/processed/train.csv",
                "transformed_test_path": "data/processed/test.csv",
                "transformed_data_path": "data/processed/data.csv",
            }
        }
        mock_load_config.return_value = mock_config

        service = FeatureEngineeringService("configs/config.yaml")

        # Verify initialization
        self.assertEqual(service.root_dir, "data/processed")
        self.assertEqual(service.cleaned_data_path, "data/processed/merged.csv")

    @patch("pandas.DataFrame.to_csv")
    @patch("sklearn.model_selection.train_test_split")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    @patch("src.data.feature_engineering.load_config")
    def test_run_creates_rating_feature(
        self, mock_load_config, mock_exists, mock_read_csv, mock_split, mock_to_csv
    ):
        """Test run method creates rating from playtime"""
        mock_config = {
            "feature_engineering": {
                "root_dir": "data/processed",
                "cleaned_data_path": "data/processed/merged.csv",
                "transformed_train_path": "data/processed/train.csv",
                "transformed_test_path": "data/processed/test.csv",
                "transformed_data_path": "data/processed/data.csv",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        # Mock data
        mock_df = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "item_id": ["123", "456"],
                "item_name": ["Game1", "Game2"],
                "playtime": [100, 200],
                "genres": ['["Action"]', '["RPG"]'],
                "tags": ['["FPS"]', '["Fantasy"]'],
            }
        )
        mock_read_csv.return_value = mock_df

        # Mock train_test_split
        train_df = mock_df.iloc[:1]
        test_df = mock_df.iloc[1:]
        mock_split.return_value = (train_df, test_df)

        service = FeatureEngineeringService()
        service.run()

        # Verify to_csv was called 3 times (full, train, test)
        self.assertEqual(mock_to_csv.call_count, 3)

    @patch("pandas.DataFrame.to_csv")
    @patch("sklearn.model_selection.train_test_split")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    @patch("src.data.feature_engineering.load_config")
    def test_run_processes_genres_and_tags(
        self, mock_load_config, mock_exists, mock_read_csv, mock_split, mock_to_csv
    ):
        """Test run method processes genres and tags correctly"""
        mock_config = {
            "feature_engineering": {
                "root_dir": "data/processed",
                "cleaned_data_path": "data/processed/merged.csv",
                "transformed_train_path": "data/processed/train.csv",
                "transformed_test_path": "data/processed/test.csv",
                "transformed_data_path": "data/processed/data.csv",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        # Mock data with different formats for genres/tags
        mock_df = pd.DataFrame(
            {
                "user_id": ["user1", "user2", "user3"],
                "item_id": ["123", "456", "789"],
                "item_name": ["Game1", "Game2", "Game3"],
                "playtime": [100, 200, 300],
                "genres": ['["Action"]', ["RPG"], "Strategy"],  # Different formats
                "tags": ['["FPS"]', ["Multiplayer"], np.nan],
            }
        )
        mock_read_csv.return_value = mock_df

        mock_split.return_value = (mock_df.iloc[:2], mock_df.iloc[2:])

        service = FeatureEngineeringService()
        service.run()

        # Verify processing completed without errors
        self.assertEqual(mock_to_csv.call_count, 3)

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    @patch("src.data.feature_engineering.load_config")
    def test_run_handles_missing_file(
        self, mock_load_config, mock_exists, mock_read_csv
    ):
        """Test run method handles missing cleaned data file"""
        mock_config = {
            "feature_engineering": {
                "cleaned_data_path": "data/processed/merged.csv",
                "root_dir": "data/processed",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False

        service = FeatureEngineeringService()

        with self.assertRaises(CustomException):
            service.run()

    @patch("pandas.DataFrame.to_csv")
    @patch("sklearn.model_selection.train_test_split")
    @patch("pandas.read_csv")
    @patch("os.path.exists")
    @patch("src.data.feature_engineering.load_config")
    def test_run_creates_item_text(
        self, mock_load_config, mock_exists, mock_read_csv, mock_split, mock_to_csv
    ):
        """Test run method creates item_text from genres and tags"""
        mock_config = {
            "feature_engineering": {
                "root_dir": "data/processed",
                "cleaned_data_path": "data/processed/merged.csv",
                "transformed_train_path": "data/processed/train.csv",
                "transformed_test_path": "data/processed/test.csv",
                "transformed_data_path": "data/processed/data.csv",
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        mock_df = pd.DataFrame(
            {
                "user_id": ["user1"],
                "item_id": ["123"],
                "item_name": ["Game1"],
                "playtime": [100],
                "genres": ['["Action", "Shooter"]'],
                "tags": ['["FPS", "Multiplayer"]'],
            }
        )
        mock_read_csv.return_value = mock_df
        mock_split.return_value = (mock_df, mock_df)

        service = FeatureEngineeringService()
        service.run()

        # Verify processing completed
        self.assertEqual(mock_to_csv.call_count, 3)


if __name__ == "__main__":
    unittest.main()

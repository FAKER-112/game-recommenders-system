"""
Unit Tests for Training Pipeline
==================================

This test suite provides comprehensive unit testing for the TrainingPipeline class
in `src/pipeline/train_pipeline.py`, which orchestrates the end-to-end training workflow
by coordinating data loading, cleaning, and model training services.

Test Coverage:
-------------

**Initialization Tests (2 tests)**
- test_init_default_config: Tests initialization with default config path ('configs/pipeline_params.yaml')
- test_init_custom_config: Tests initialization with custom config path provided by user

**Training Workflow Tests (2 tests)**
- test_train_model_success_flow: Tests the complete training pipeline execution
  * Verifies LoadDataService, CleanDataService, and ModelTrainingService are instantiated
  * Verifies all services are called with correct config paths
  * Validates proper logging of pipeline stages
  * Tests with specific model name ('tfrs') passed through to training service

- test_train_model_exception_handling: Tests CustomException is raised when services fail
  * Simulates LoadDataService failure
  * Verifies exception propagation with correct error message

Pipeline Flow Tested:
--------------------
1. **Configuration Loading**: Pipeline config contains paths to data and training configs
2. **Service Instantiation**: Creates LoadDataService, CleanDataService, ModelTrainingService
3. **Sequential Execution**: Services run in order (load → clean → train)
4. **Model Routing**: Model name parameter passed to training service for model-specific training
5. **Error Handling**: Exceptions wrapped in CustomException for consistent error handling

Mocking Strategy:
----------------
All tests mock external dependencies to ensure:
- Fast execution (no actual data loading or model training)
- Isolation from config files and data files
- Controlled testing of service coordination logic

Mocked components:
- load_config: Returns mock configuration dictionaries
- LoadDataService, CleanDataService, ModelTrainingService: Mocked to verify orchestration
- logger: Prevents actual logging while allowing verification of log calls

Usage:
------
Run all tests:
    python -m pytest tests/test_train_pipeline.py -v

Run specific test:
    python -m pytest tests/test_train_pipeline.py::TestTrainingPipeline::test_train_model_success_flow -v

Run with unittest:
    python -m unittest tests.test_train_pipeline
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call

# Add project root to the Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.pipeline.train_pipeline import TrainingPipeline
from src.utils.exception import CustomException


class TestTrainingPipeline(unittest.TestCase):

    @patch("src.pipeline.train_pipeline.load_config")
    def test_init_default_config(self, mock_load_config):
        """
        Tests if the TrainingPipeline initializes correctly with the default config path.
        """
        mock_config = {
            "Training_pipeline": {
                "data_config_path": "configs/config.yaml",
                "training_config_path": "configs/model_params.yaml",
            }
        }
        mock_load_config.return_value = mock_config

        pipeline = TrainingPipeline()

        mock_load_config.assert_called_once_with("configs/pipeline_params.yaml")
        self.assertEqual(pipeline.load_data_config, "configs/config.yaml")
        self.assertEqual(pipeline.training_config_path, "configs/model_params.yaml")

    @patch("src.pipeline.train_pipeline.load_config")
    def test_init_custom_config(self, mock_load_config):
        """
        Tests if the TrainingPipeline initializes correctly with a custom config path.
        """
        mock_config = {
            "Training_pipeline": {
                "data_config_path": "custom/data.yaml",
                "training_config_path": "custom/model.yaml",
            }
        }
        mock_load_config.return_value = mock_config
        custom_path = "custom/pipeline.yaml"

        pipeline = TrainingPipeline(config_path=custom_path)

        mock_load_config.assert_called_once_with(custom_path)
        self.assertEqual(pipeline.load_data_config, "custom/data.yaml")
        self.assertEqual(pipeline.training_config_path, "custom/model.yaml")

    @patch("src.pipeline.train_pipeline.ModelTrainingService")
    @patch("src.pipeline.train_pipeline.CleanDataService")
    @patch("src.pipeline.train_pipeline.LoadDataService")
    @patch("src.pipeline.train_pipeline.load_config")
    def test_train_model_success_flow(
        self, mock_load_config, mock_load_data, mock_clean_data, mock_model_training
    ):
        """
        Tests the successful execution flow of the train_model method.
        """
        mock_load_config.return_value = {
            "Training_pipeline": {
                "data_config_path": "configs/config.yaml",
                "training_config_path": "configs/model_params.yaml",
            }
        }

        pipeline = TrainingPipeline()
        pipeline.logger = (
            MagicMock()
        )  # Mock logger to prevent actual logging and allow call assertions
        pipeline.train_model(model_name="tfrs")

        # Verify services are instantiated with correct configs
        mock_load_data.assert_called_once_with("configs/config.yaml")
        mock_clean_data.assert_called_once_with("configs/config.yaml")
        mock_model_training.assert_called_once_with("configs/model_params.yaml")

        # Verify the run method of each service is called
        mock_load_data.return_value.run.assert_called_once()
        mock_clean_data.return_value.run.assert_called_once()
        mock_model_training.return_value.run.assert_called_once_with("tfrs")

        # Verify logging calls
        self.assertIn(
            call("Training pipeline started"), pipeline.logger.info.call_args_list
        )
        self.assertIn(call("Training model tfrs"), pipeline.logger.info.call_args_list)

    @patch("src.pipeline.train_pipeline.LoadDataService")
    @patch("src.pipeline.train_pipeline.load_config")
    def test_train_model_exception_handling(self, mock_load_config, mock_load_data):
        """
        Tests that a CustomException is raised if a service fails.
        """
        mock_load_config.return_value = {"Training_pipeline": {}}
        error_message = "Failed to load data"
        mock_load_data.return_value.run.side_effect = Exception(error_message)

        pipeline = TrainingPipeline()

        with self.assertRaises(CustomException) as cm:
            pipeline.train_model()

        self.assertIn(error_message, str(cm.exception))


if __name__ == "__main__":
    unittest.main()

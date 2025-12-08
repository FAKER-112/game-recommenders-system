import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.pipeline.train_pipeline import TrainingPipeline
from src.utils.exception import CustomException

class TestTrainingPipeline(unittest.TestCase):

    @patch('src.pipeline.train_pipeline.load_config')
    def test_init_default_config(self, mock_load_config):
        """
        Tests if the TrainingPipeline initializes correctly with the default config path.
        """
        mock_config = {
            'Training_pipeline': {
                'data_config_path': 'configs/config.yaml',
                'training_config_path': 'configs/model_params.yaml'
            }
        }
        mock_load_config.return_value = mock_config

        pipeline = TrainingPipeline()

        mock_load_config.assert_called_once_with('configs/pipeline_params.yaml')
        self.assertEqual(pipeline.load_data_config, 'configs/config.yaml')
        self.assertEqual(pipeline.training_config_path, 'configs/model_params.yaml')

    @patch('src.pipeline.train_pipeline.load_config')
    def test_init_custom_config(self, mock_load_config):
        """
        Tests if the TrainingPipeline initializes correctly with a custom config path.
        """
        mock_config = {
            'Training_pipeline': {
                'data_config_path': 'custom/data.yaml',
                'training_config_path': 'custom/model.yaml'
            }
        }
        mock_load_config.return_value = mock_config
        custom_path = 'custom/pipeline.yaml'

        pipeline = TrainingPipeline(config_path=custom_path)

        mock_load_config.assert_called_once_with(custom_path)
        self.assertEqual(pipeline.load_data_config, 'custom/data.yaml')
        self.assertEqual(pipeline.training_config_path, 'custom/model.yaml')

    @patch('src.pipeline.train_pipeline.ModelTrainingService')
    @patch('src.pipeline.train_pipeline.CleanDataService')
    @patch('src.pipeline.train_pipeline.LoadDataService')
    @patch('src.pipeline.train_pipeline.load_config')
    def test_train_model_success_flow(self, mock_load_config, mock_load_data, mock_clean_data, mock_model_training):
        """
        Tests the successful execution flow of the train_model method.
        """
        mock_load_config.return_value = {
            'Training_pipeline': {
                'data_config_path': 'configs/config.yaml',
                'training_config_path': 'configs/model_params.yaml'
            }
        }

        pipeline = TrainingPipeline()
        pipeline.logger = MagicMock() # Mock logger to prevent actual logging and allow call assertions
        pipeline.train_model(model_name='tfrs')

        # Verify services are instantiated with correct configs
        mock_load_data.assert_called_once_with('configs/config.yaml')
        mock_clean_data.assert_called_once_with('configs/config.yaml')
        mock_model_training.assert_called_once_with('configs/model_params.yaml')

        # Verify the run method of each service is called
        mock_load_data.return_value.run.assert_called_once()
        mock_clean_data.return_value.run.assert_called_once()
        mock_model_training.return_value.run.assert_called_once_with('tfrs')

        # Verify logging calls
        self.assertIn(call('Training pipeline started'), pipeline.logger.info.call_args_list)
        self.assertIn(call('Training model tfrs'), pipeline.logger.info.call_args_list)

    @patch('src.pipeline.train_pipeline.LoadDataService')
    @patch('src.pipeline.train_pipeline.load_config')
    def test_train_model_exception_handling(self, mock_load_config, mock_load_data):
        """
        Tests that a CustomException is raised if a service fails.
        """
        mock_load_config.return_value = {'Training_pipeline': {}}
        error_message = "Failed to load data"
        mock_load_data.return_value.run.side_effect = Exception(error_message)

        pipeline = TrainingPipeline()

        with self.assertRaises(CustomException) as cm:
            pipeline.train_model()
        
        self.assertIn(error_message, str(cm.exception))

if __name__ == '__main__':
    unittest.main()

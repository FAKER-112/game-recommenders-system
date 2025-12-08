import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np
import tensorflow as tf

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.exception import CustomException

class TestPredictionPipeline(unittest.TestCase):

    def setUp(self):
        """Set up common mock objects for tests."""
        self.mock_config = {
            'model_training': {
                'root_dir': 'artifacts/models',
                'context_dir': 'artifacts/context'
            }
        }
        # Mocking dependencies that are almost always needed
        self.load_config_patcher = patch('src.pipeline.predict_pipeline.load_config', return_value=self.mock_config)
        self.logger_patcher = patch('src.pipeline.predict_pipeline.logger', MagicMock())
        self.os_path_exists_patcher = patch('os.path.exists', return_value=True)

        self.mock_load_config = self.load_config_patcher.start()
        self.mock_logger = self.logger_patcher.start()
        self.mock_os_path_exists = self.os_path_exists_patcher.start()

    def tearDown(self):
        """Stop all patchers."""
        self.load_config_patcher.stop()
        self.logger_patcher.stop()
        self.os_path_exists_patcher.stop()
        patch.stopall()

    @patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts')
    @patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model')
    def test_init_success(self, mock_load_model, mock_load_artifacts):
        """Test successful initialization of the PredictionPipeline."""
        pipeline = PredictionPipeline(model_name="tfrs", config_path="dummy/path.yaml")
        
        self.mock_load_config.assert_called_once_with("dummy/path.yaml")
        self.assertEqual(pipeline.model_name, "tfrs")
        self.assertEqual(pipeline.root_dir, 'artifacts/models')
        self.assertEqual(pipeline.context_dir, 'artifacts/context')
        mock_load_model.assert_called_once()
        mock_load_artifacts.assert_called_once()

    @patch('src.pipeline.predict_pipeline.load_config', side_effect=Exception("Config error"))
    def test_init_failure(self, mock_load_config_fail):
        """Test initialization failure raises CustomException."""
        with self.assertRaises(CustomException) as cm:
            PredictionPipeline()
        self.assertIn("Config error", str(cm.exception))

    @patch('tensorflow.keras.models.load_model')
    def test_load_model_mf(self, mock_load_keras_model):
        """Test loading the Matrix Factorization model."""
        pipeline = PredictionPipeline(model_name="mf")
        mock_load_keras_model.assert_called_once_with('artifacts/models/mf_model.h5')
        self.mock_logger.info.assert_called_with("Loaded MF model from artifacts/models/mf_model.h5")

    @patch('tensorflow.saved_model.load')
    def test_load_model_tfrs(self, mock_tf_load):
        """Test loading the TFRS model and indices."""
        pipeline = PredictionPipeline(model_name="tfrs")
        
        calls = [
            call('artifacts/models/tfrs_retrieval_index'),
            call('artifacts/models/tfrs_item_index')
        ]
        mock_tf_load.assert_has_calls(calls)
        self.mock_logger.info.assert_any_call("Loaded TFRS retrieval index from artifacts/models/tfrs_retrieval_index")
        self.mock_logger.info.assert_any_call("Loaded TFRS item index from artifacts/models/tfrs_item_index")

    def test_load_model_tfrs_file_not_found(self):
        """Test TFRS model loading when index file is not found."""
        self.mock_os_path_exists.return_value = False
        with self.assertRaises(CustomException):
            PredictionPipeline(model_name="tfrs")

    def test_load_model_invalid_name(self):
        """Test loading an invalid model name raises an exception."""
        with self.assertRaises(CustomException) as cm:
            # We need to bypass the __init__ mocks to test _load_model directly
            pipeline = PredictionPipeline.__new__(PredictionPipeline)
            pipeline.model_name = "invalid_model"
            pipeline.root_dir = "artifacts/models"
            pipeline.logger = self.mock_logger
            pipeline._load_model()
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    @patch('pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_artifacts_mf(self, mock_file, mock_pickle_load):
        """Test loading artifacts for the MF model."""
        # We need to re-patch _load_model because it's called in __init__
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'):
            pipeline = PredictionPipeline(model_name="mf")
        
        user_encoder_path = os.path.join('artifacts/context', "mf_user_encoder.pkl")
        item_encoder_path = os.path.join('artifacts/context', "mf_item_encoder.pkl")
        
        calls = [call(user_encoder_path, 'rb'), call(item_encoder_path, 'rb')]
        mock_file.assert_has_calls(calls, any_order=True)
        self.assertEqual(mock_pickle_load.call_count, 2)

    @patch('pandas.read_csv')
    def test_load_artifacts_tfrs(self, mock_read_csv):
        """Test loading artifacts for the TFRS model."""
        mock_df = pd.DataFrame({'item_name': ['game1']})
        mock_read_csv.return_value = mock_df
        
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'):
            pipeline = PredictionPipeline(model_name="tfrs")
        
        candidates_path = os.path.join('artifacts/context', "tfrs_candidates.csv")
        mock_read_csv.assert_called_once_with(candidates_path)
        pd.testing.assert_frame_equal(pipeline.unique_items_df, mock_df)
        self.mock_logger.info.assert_called_with(f"Loaded {len(mock_df)} candidates from {candidates_path}")

    @patch('numpy.load')
    @patch('pandas.read_csv')
    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_artifacts_autoencoder(self, mock_file, mock_json_load, mock_read_csv, mock_np_load):
        """Test loading artifacts for the Autoencoder model."""
        mock_read_csv.return_value = pd.DataFrame({0: ['item1'], 1: [0]})
        mock_json_load.return_value = ['item1', 'item2']
        
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'):
            pipeline = PredictionPipeline(model_name="autoencoder")
        
        mock_np_load.assert_called_once_with(os.path.join('artifacts/context', 'autoencoder_X.npy'))
        mock_read_csv.assert_called_once_with(os.path.join('artifacts/context', 'autoencoder_indices.csv'))
        mock_file.assert_called_with(os.path.join('artifacts/context', 'autoencoder_itemlist.json'), 'r')
        mock_json_load.assert_called_once()

    @patch('src.pipeline.predict_pipeline.PredictionPipeline._recommend_tfrs')
    def test_recommend_dispatches_to_tfrs(self, mock_recommend_tfrs):
        """Test that recommend() calls the correct method for TFRS."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="tfrs")
        pipeline.recommend(user_id="user1", n_rec=5)
        mock_recommend_tfrs.assert_called_once_with("user1", 5)

    @patch('src.pipeline.predict_pipeline.PredictionPipeline._recommend_mf')
    def test_recommend_dispatches_to_mf(self, mock_recommend_mf):
        """Test that recommend() calls the correct method for MF."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="mf")
        pipeline.recommend(user_id="user1", n_rec=5)
        mock_recommend_mf.assert_called_once_with("user1", 5)

    def test_recommend_not_implemented(self):
        """Test that recommend() returns empty list for not implemented models."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="autoencoder")
        result = pipeline.recommend(user_id="user1")
        self.assertEqual(result, [])
        self.mock_logger.warning.assert_called_with("Recommend not implemented for autoencoder")

    def test_recommend_tfrs(self):
        """Test TFRS recommendation logic."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="tfrs")
        
        # Mock the retrieval index
        mock_index = MagicMock()
        titles_tensor = tf.constant([['rec1', 'rec2', 'rec3']], dtype=tf.string)
        mock_index.return_value = (None, titles_tensor)
        pipeline.retrieval_index = mock_index
        
        recs = pipeline.recommend(user_id="user1", n_rec=2)
        
        mock_index.assert_called_once()
        self.assertEqual(recs, ['rec1', 'rec2'])

    def test_recommend_mf(self):
        """Test MF recommendation logic."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="mf")

        # Mock encoders
        mock_user_encoder = MagicMock()
        mock_user_encoder.classes_ = np.array(['user1', 'user2'])
        mock_user_encoder.transform.return_value = np.array([0])
        pipeline.user_encoder = mock_user_encoder

        mock_item_encoder = MagicMock()
        mock_item_encoder.classes_ = np.array(['item1', 'item2', 'item3'])
        mock_item_encoder.inverse_transform.return_value = np.array(['item3', 'item1'])
        pipeline.item_encoder = mock_item_encoder
        
        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.5, 0.9]]) # Predictions for 3 items
        pipeline.model = mock_model
        
        recs = pipeline.recommend(user_id="user1", n_rec=2)
        
        mock_user_encoder.transform.assert_called_with(['user1'])
        self.assertEqual(recs, ['item3', 'item1'])
        
    def test_recommend_mf_user_not_found(self):
        """Test MF recommendation for a user not in the training data."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="mf")
        mock_user_encoder = MagicMock()
        mock_user_encoder.classes_ = np.array(['user2'])
        pipeline.user_encoder = mock_user_encoder
        
        recs = pipeline.recommend(user_id="user1")
        
        self.assertEqual(recs, [])
        self.mock_logger.warning.assert_called_with("User user1 not found in training data.")

    @patch('src.pipeline.predict_pipeline.PredictionPipeline._get_similar_items_tfrs')
    def test_get_similar_items_dispatches_to_tfrs(self, mock_similar_tfrs):
        """Test that get_similar_items() calls the correct method for TFRS."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="tfrs")
        pipeline.get_similar_items(item_name="item1", k=5)
        mock_similar_tfrs.assert_called_once_with("item1", 5)

    @patch('src.pipeline.predict_pipeline.PredictionPipeline._get_similar_items_autoencoder')
    def test_get_similar_items_dispatches_to_autoencoder(self, mock_similar_ae):
        """Test that get_similar_items() calls the correct method for Autoencoder."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="autoencoder")
        pipeline.get_similar_items(item_name="item1", k=5)
        mock_similar_ae.assert_called_once_with("item1", 5)

    def test_get_similar_items_tfrs(self):
        """Test TFRS similar items logic."""
        with patch('src.pipeline.predict_pipeline.PredictionPipeline._load_model'), \
             patch('src.pipeline.predict_pipeline.PredictionPipeline._load_artifacts'):
            pipeline = PredictionPipeline(model_name="tfrs")
        
        # Mock item index
        mock_item_index = MagicMock()
        similar_titles_tensor = tf.constant([['item1', 'sim1', 'sim2']], dtype=tf.string)
        mock_item_index.return_value = (None, similar_titles_tensor)
        pipeline.item_index = mock_item_index
        
        # Mock candidates df
        pipeline.unique_items_df = pd.DataFrame({
            'item_name': ['item1', 'sim1', 'sim2'],
            'item_text': ['text1', 'text_sim1', 'text_sim2']
        })
        
        similar = pipeline.get_similar_items(item_name="item1", k=2)
        
        mock_item_index.assert_called_once()
        self.assertEqual(similar, ['sim1', 'sim2'])

if __name__ == '__main__':
    unittest.main()
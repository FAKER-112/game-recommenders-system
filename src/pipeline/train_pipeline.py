import os 
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.models.train_model import ModelTrainingService
from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.utils import load_config 
from src.data.load_data import LoadDataService
from src.data.clean_data import CleanDataService
from src.data.feature_engineering import FeatureEngineeringService

class TrainingPipeline:
    def __init__(self, config_path:str ='configs/pipeline_params.yaml'):
        self.config= load_config(config_path)
        self.logger = logger
        pipeline_config= self.config.get('Training_pipeline',{})
        self.load_data_config= pipeline_config.get("data_config_path","configs/config.yaml")
        self.training_config_path= pipeline_config.get("training_config_path","configs/model_params.yaml")

    def train_model(self, model_name:str='autoencoder'):
        try:
            self.logger.info("Training pipeline started")
            self.logger.info('Loading data')
            load_data=LoadDataService(self.load_data_config)
            load_data.run()
            self.logger.info('loading data completed')
            self.logger.info('Cleaning data')
            clean_data=CleanDataService(self.load_data_config)
            clean_data.run()
            self.logger.info('Cleaning data completed')
            self.logger.info('Feature engineering started')
            feature_engineering=FeatureEngineeringService(self.load_data_config)
            feature_engineering.run()
            self.logger.info('Feature engineering completed')
            self.logger.info(f'Training model {model_name}')
            model_trainer=ModelTrainingService(self.training_config_path)
            model_trainer.run(model_name)
            

            pass
        except Exception as e:    
            raise CustomException(e)   
        
if __name__=="__main__":
    training_pipeline=TrainingPipeline()
    training_pipeline.train_model('mf')



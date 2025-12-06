import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from src.models.evaluate_model import ModelEvaluationService
from src.models.train_model import ModelTrainingService
from src.utils.exception import CustomException
from src.utils.logger import logger


def run_evaluation():
    try:
        evaluator = ModelEvaluationService()
        evaluator.run(model_name='mf')
    except Exception as e:
        raise CustomException(e)


if __name__ == "__main__":
    run_evaluation()


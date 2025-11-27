import os
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from src.models.train_model import ModelTrainingService
from src.models.evaluate_model import ModelEvaluationService
from src.utils.exception import CustomException
from src.utils.logger import logger


class PredictionPipeline:
    def __init__(self, model_name: str  = 'tfrs'):
        self.model_name = model_name
        self.root_dir = model_cfg.get("root_dir", "artifacts/models")
        self._load_model()
    def _load_model(self):
        if self.model_name =='mf' or self.model_name =='autoencoder':
            self.model = load_model(os.path.join(self.root_dir, f"{self.model_name}.h5"))
        elif self.model_name =='tfrs':
            self.model= tf.saved_model.load(os.path.join(self.root_dir, f"{self.model_name}"))

    def prepare_auto_encoder(self):
        pass

    def prepare_mf(self):
        pass
    def prepare_tfrs(self):
        pass
    def recommend(self):
        if self.model_name == 'autoencoder' :
            autoencoder = self.model
            encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)
            X= np.load('autoencoder_X.npz')


        elif self.model_name == 'mf':
            pass
        elif self.model_name ==  'tfrs':
            pass
    def get_similar_items(self, item:str) -> list:
        if self.model_name == 'autoencoder' :
            autoencoder = self.model
            encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('bottleneck').output)
            X= np.load('autoencoder_X.npz')
            indices = pd.read_csv('autoencoder_indices.csv')
            embeddings = encoder.predict(X, verbose=0)
            

            # Calculate similarity using the new neural fingerprints
            neural_sim = cosine_similarity(neural_embeddings)
            with open('autoencoder_itemlist.json', 'r') as f:
                global_item_names = json.load(f)

            def get_neural_recommendations(title):
                try:
                    idx = indices[title]
                except KeyError:
                    return "Game not found."

                # Get similarity scores
                sim_scores = list(enumerate(neural_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]

                # Get the game indices
                game_indices = [i[0] for i in sim_scores]

                # Return the top 10 most similar games using the global list
                return [global_item_names[i] for i in game_indices]

            recommendation = get_neural_recommendations(item)

        elif self.model_name == 'mf':
            pass
        elif self.model_name ==  'tfrs':
            pass
        return recommendation

if __name__ == "__main__":
    PredictionPipeline()

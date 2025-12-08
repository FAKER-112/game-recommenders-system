import os 
import sys
from flask import Flask, request, jsonify, render_template
import numpy as np
from src.pipeline.predict_pipeline import PredictionPipeline


app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "API is running"})
@app.route('/recommend_user', methods=['POST'])
def recomend_user():
    input= request.get_json()
    pipeline= PredictionPipeline(input['model_name'])
    user_recommendations = pipeline.recomend(input['user_id'], input['n_rec'])
    return jsonify(user_recommendations)



@app.route('/recommend_item', methods=['POST'])
def recomend_item():
    input= request.get_json()
    pipeline= PredictionPipeline(input['model_name'])
    similar_items = pipeline.get_similar_items(input['item_name'], input['k'])
    return jsonify(similar_items)
@app.route('/health')
def health():
    return jsonify({"status": "API is running"})
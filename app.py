"""
Flask API Server Module

This script defines the Flask application that serves as the backend API for the Game Recommender System.
It provides endpoints for data retrieval, system health checks, and generating recommendations using trained models.

Logic of Operation:
1.  **Server Initialization**: Configures the Flask app and CORS to allow frontend communication.
2.  **Data Preloading**:
    - Loads game metadata from `steam_games.json.gz` into memory (`GAMES_DATA`) for fast lookup.
    - Loads active user IDs from processed CSVs into memory (`USER_IDS`).
3.  **Model Management**:
    - Uses a `MODEL_CACHE` to store loaded `PredictionPipeline` instances.
    - Lazily loads models (TFRS, MF, Autoencoder) upon first request to optimize startup time.
4.  **API Endpoints**:
    - **Data**: Exposes endpoints to fetch user lists and game details with pagination and search.
    - **Inference**: Routes recommendation requests to the appropriate model pipeline (`/recommend_user`, `/recommend_item`).
    - **Batch Processing**: Handles batch recommendation requests for multiple users.
5.  **Error Handling**: Implements global error handling decorators to ensure consistent JSON error responses.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime
from functools import wraps
import traceback
import gzip
import ast

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.pipeline.predict_pipeline import PredictionPipeline
from src.utils.exception import CustomException
from src.utils.logger import logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for game data and user IDs
GAMES_DATA = []
USER_IDS = []
ITEMS_PER_PAGE = 10

# Cache for model pipelines to avoid reloading
MODEL_CACHE = {}


def load_games_data():
    """Load and preprocess game data from steam_games.json.gz"""
    global GAMES_DATA
    try:
        logger.info("Loading game data from steam_games.json.gz...")

        # Path to the gzipped game data
        games_path = "data/raw/steam_games.json.gz"

        if not os.path.exists(games_path):
            logger.error(f"Game data file not found: {games_path}")
            return

        games = []
        with gzip.open(games_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    game_dict = ast.literal_eval(line.strip())

                    # Extract and format the fields we need
                    game = {
                        "id": str(game_dict.get("id", "")),
                        "title": game_dict.get("title", game_dict.get("app_name", "")),
                        "app_name": game_dict.get("app_name", ""),
                        "genres": game_dict.get("genres", []),
                        "tags": game_dict.get("tags", []),
                        "price": game_dict.get("price", 0.0),
                        "discount_price": game_dict.get("discount_price", None),
                        "release_date": game_dict.get("release_date", ""),
                        "developer": game_dict.get("developer", ""),
                        "publisher": game_dict.get("publisher", ""),
                        "url": game_dict.get("url", ""),
                        "early_access": game_dict.get("early_access", False),
                        "specs": game_dict.get("specs", []),
                        "sentiment": game_dict.get("sentiment", ""),
                    }

                    # Clean up price values
                    if isinstance(game["price"], str):
                        if game["price"].lower() in ["free to play", "free"]:
                            game["price"] = 0.0
                        else:
                            try:
                                game["price"] = float(game["price"])
                            except:
                                game["price"] = 0.0

                    if isinstance(game["discount_price"], str):
                        try:
                            game["discount_price"] = float(game["discount_price"])
                        except:
                            game["discount_price"] = None

                    games.append(game)

                except Exception as e:
                    logger.warning(f"Error parsing game data line: {e}")
                    continue

        GAMES_DATA = games
        logger.info(f"Successfully loaded {len(GAMES_DATA)} games")

    except Exception as e:
        logger.error(f"Failed to load game data: {e}")
        raise CustomException(e)


def load_user_ids():
    """Load and preprocess user IDs from the transformed data"""
    global USER_IDS
    try:
        logger.info("Loading user IDs from processed data...")

        # Try to load from processed data
        processed_data_path = "data/processed/test.csv"

        if os.path.exists(processed_data_path):
            df = pd.read_csv(processed_data_path)
            if "user_id" in df.columns:
                USER_IDS = df["user_id"].astype(str).unique().tolist()
                logger.info(f"Loaded {len(USER_IDS)} user IDs from processed data")
            else:
                logger.warning(f"'user_id' column not found in {processed_data_path}")
                # Fallback to dummy users
                USER_IDS = [
                    "76561197970982479",
                    "76561198001103300",
                    "76561198002935466",
                ]
        else:
            logger.warning(f"Processed data not found at {processed_data_path}")
            # Fallback to dummy users
            USER_IDS = ["76561197970982479", "76561198001103300", "76561198002935466"]

    except Exception as e:
        logger.error(f"Failed to load user IDs: {e}")
        # Fallback to dummy users
        USER_IDS = ["76561197970982479", "76561198001103300", "76561198002935466"]


def get_pipeline(model_name: str) -> PredictionPipeline:
    """
    Get or create a pipeline instance with caching.

    Args:
        model_name: Name of the model ('tfrs', 'mf', or 'autoencoder')

    Returns:
        PredictionPipeline instance
    """
    if model_name not in MODEL_CACHE:
        logger.info(f"Loading {model_name} pipeline...")
        try:
            MODEL_CACHE[model_name] = PredictionPipeline(model_name=model_name)
            logger.info(f"{model_name} pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load {model_name} pipeline: {e}")
            raise
    return MODEL_CACHE[model_name]


def validate_input(required_fields):
    """
    Decorator to validate request input fields.

    Args:
        required_fields: List of required field names
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                data = request.get_json()

                if not data:
                    return (
                        jsonify({"error": "No JSON data provided", "status": "error"}),
                        400,
                    )

                # Check for missing fields
                missing_fields = [
                    field for field in required_fields if field not in data
                ]
                if missing_fields:
                    return (
                        jsonify(
                            {
                                "error": f'Missing required fields: {", ".join(missing_fields)}',
                                "status": "error",
                            }
                        ),
                        400,
                    )

                return f(*args, **kwargs)

            except Exception as e:
                logger.error(f"Validation error: {e}")
                return (
                    jsonify({"error": "Invalid request format", "status": "error"}),
                    400,
                )

        return decorated_function

    return decorator


def handle_errors(f):
    """
    Decorator to handle errors consistently across endpoints.
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except CustomException as e:
            logger.error(f"Custom exception in {f.__name__}: {e}")
            return (
                jsonify(
                    {"error": str(e), "status": "error", "type": "CustomException"}
                ),
                500,
            )
        except ValueError as e:
            logger.error(f"Value error in {f.__name__}: {e}")
            return (
                jsonify({"error": str(e), "status": "error", "type": "ValueError"}),
                400,
            )
        except FileNotFoundError as e:
            logger.error(f"File not found in {f.__name__}: {e}")
            return (
                jsonify(
                    {
                        "error": "Model or required files not found. Please ensure models are trained.",
                        "status": "error",
                        "type": "FileNotFoundError",
                        "details": str(e),
                    }
                ),
                404,
            )
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            logger.error(traceback.format_exc())
            return (
                jsonify(
                    {
                        "error": "An unexpected error occurred",
                        "status": "error",
                        "type": type(e).__name__,
                        "details": str(e),
                    }
                ),
                500,
            )

    return decorated_function


# ============================================================================
# NEW ENDPOINTS FOR FRONTEND DATA
# ============================================================================


@app.route("/api/userlist", methods=["GET"])
@handle_errors
def get_userlist():
    """
    Get list of available user IDs.

    Response JSON:
    {
        "status": "success",
        "users": ["user1", "user2", ...],
        "count": 100,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    return (
        jsonify(
            {
                "status": "success",
                "users": USER_IDS,
                "count": len(USER_IDS),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.route("/api/gamedata", methods=["GET"])
@handle_errors
def get_gamedata():
    """
    Get paginated game data.

    Query Parameters:
    - start: Starting index (default: 0)
    - end: Ending index (default: 10)
    - search: Optional search term to filter games

    Response JSON:
    {
        "status": "success",
        "games": [{game1}, {game2}, ...],
        "total_games": 1000,
        "start": 0,
        "end": 10,
        "has_more": true,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    try:
        start = request.args.get("start", default=0, type=int)
        end = request.args.get("end", default=ITEMS_PER_PAGE, type=int)
        search_term = request.args.get("search", default="", type=str)

        # Validate indices
        if start < 0:
            start = 0
        if end < start:
            end = start + ITEMS_PER_PAGE

        # Apply search filter if provided
        if search_term:
            search_lower = search_term.lower()
            filtered_games = []
            for game in GAMES_DATA:
                title = game.get("title", "").lower()
                app_name = game.get("app_name", "").lower()
                genres = " ".join(game.get("genres", [])).lower()
                tags = " ".join(game.get("tags", [])).lower()

                if (
                    search_lower in title
                    or search_lower in app_name
                    or search_lower in genres
                    or search_lower in tags
                ):
                    filtered_games.append(game)
        else:
            filtered_games = GAMES_DATA

        # Get the requested slice
        total_games = len(filtered_games)
        games_slice = filtered_games[start:end]

        return (
            jsonify(
                {
                    "status": "success",
                    "games": games_slice,
                    "total_games": total_games,
                    "start": start,
                    "end": min(end, total_games),
                    "has_more": end < total_games,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error in get_gamedata: {e}")
        return (
            jsonify(
                {
                    "error": "Failed to retrieve game data",
                    "status": "error",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/api/game/<game_id>", methods=["GET"])
@handle_errors
def get_game_by_id(game_id):
    """
    Get detailed information for a specific game by ID.

    Response JSON:
    {
        "status": "success",
        "game": {game_details},
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    try:
        # Try to find by ID
        for game in GAMES_DATA:
            if str(game.get("id", "")) == str(game_id):
                return (
                    jsonify(
                        {
                            "status": "success",
                            "game": game,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    200,
                )

        # If not found by ID, try to find by title
        for game in GAMES_DATA:
            if game.get("title", "").lower() == game_id.lower():
                return (
                    jsonify(
                        {
                            "status": "success",
                            "game": game,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ),
                    200,
                )

        return (
            jsonify(
                {
                    "status": "error",
                    "error": f"Game not found: {game_id}",
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            404,
        )

    except Exception as e:
        logger.error(f"Error in get_game_by_id: {e}")
        return (
            jsonify(
                {
                    "error": "Failed to retrieve game",
                    "status": "error",
                    "details": str(e),
                }
            ),
            500,
        )


# ============================================================================
# EXISTING RECOMMENDATION ENDPOINTS (MODIFIED TO USE NEW DATA STRUCTURES)
# ============================================================================


@app.route("/")
def home():
    """Home endpoint with API information."""
    return jsonify(
        {
            "status": "success",
            "message": "Game Recommendation API is running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "userlist": "/api/userlist [GET]",
                "gamedata": "/api/gamedata [GET]",
                "game_detail": "/api/game/<id> [GET]",
                "recommend_user": "/recommend_user [POST]",
                "recommend_item": "/recommend_item [POST]",
                "batch_recommend": "/batch_recommend [POST]",
                "model_info": "/model_info/<model_name> [GET]",
                "health": "/health [GET]",
            },
            "supported_models": ["tfrs", "mf", "autoencoder"],
            "stats": {
                "total_games": len(GAMES_DATA),
                "total_users": len(USER_IDS),
                "models_loaded": list(MODEL_CACHE.keys()),
            },
        }
    )


@app.route("/health")
def health():
    """Health check endpoint."""
    models_loaded = list(MODEL_CACHE.keys())
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": models_loaded,
            "available_models": ["tfrs", "mf", "autoencoder"],
            "data_loaded": {"games": len(GAMES_DATA) > 0, "users": len(USER_IDS) > 0},
            "counts": {"games": len(GAMES_DATA), "users": len(USER_IDS)},
        }
    )


@app.route("/recommend_user", methods=["POST"])
@validate_input(["model_name", "user_id"])
@handle_errors
def recommend_user():
    """
    Generate personalized recommendations for a user.

    Request JSON:
    {
        "model_name": "tfrs",  # or "mf"
        "user_id": "76561197970982479",
        "n_rec": 10  # optional, default 10
    }

    Response JSON:
    {
        "status": "success",
        "user_id": "76561197970982479",
        "model_used": "tfrs",
        "recommendations": ["Game 1", "Game 2", ...],
        "recommendations_with_details": [{game1}, {game2}, ...],
        "count": 10,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    data = request.get_json()

    model_name = data["model_name"].lower()
    user_id = str(data["user_id"])
    n_rec = data.get("n_rec", 10)

    # Validate model name
    if model_name not in ["tfrs", "mf"]:
        return (
            jsonify(
                {
                    "error": f'Invalid model name: {model_name}. Use "tfrs" or "mf" for user recommendations.',
                    "status": "error",
                }
            ),
            400,
        )

    # Validate n_rec
    if not isinstance(n_rec, int) or n_rec < 1 or n_rec > 100:
        return (
            jsonify(
                {
                    "error": "n_rec must be an integer between 1 and 100",
                    "status": "error",
                }
            ),
            400,
        )

    logger.info(
        f"Generating {n_rec} recommendations for user {user_id} using {model_name}"
    )

    # Get pipeline and generate recommendations
    pipeline = get_pipeline(model_name)
    recommendations = pipeline.recommend(user_id=user_id, n_rec=n_rec)

    # Get full game details for each recommendation
    recommendations_with_details = []
    for game_name in recommendations:
        for game in GAMES_DATA:
            if (
                game.get("title", "").lower() == game_name.lower()
                or game.get("app_name", "").lower() == game_name.lower()
            ):
                recommendations_with_details.append(game)
                break

    if not recommendations:
        logger.warning(f"No recommendations found for user {user_id}")
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"No recommendations found for user {user_id}. User may not be in training data.",
                    "user_id": user_id,
                    "model_used": model_name,
                    "recommendations": [],
                    "recommendations_with_details": [],
                    "count": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    logger.info(f"Successfully generated {len(recommendations)} recommendations")

    return (
        jsonify(
            {
                "status": "success",
                "user_id": user_id,
                "model_used": model_name,
                "recommendations": recommendations,
                "recommendations_with_details": recommendations_with_details,
                "count": len(recommendations),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.route("/recommend_item", methods=["POST"])
@validate_input(["model_name", "item_name"])
@handle_errors
def recommend_item():
    """
    Find similar items to a given item.

    Request JSON:
    {
        "model_name": "autoencoder",  # or "tfrs"
        "item_name": "Counter-Strike",
        "k": 10  # optional, default 10
    }

    Response JSON:
    {
        "status": "success",
        "query_item": "Counter-Strike",
        "model_used": "autoencoder",
        "similar_items": ["Game 1", "Game 2", ...],
        "similar_items_with_details": [{game1}, {game2}, ...],
        "count": 10,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    data = request.get_json()

    model_name = data["model_name"].lower()
    item_name = str(data["item_name"])
    k = data.get("k", 10)

    # Validate model name
    if model_name not in ["autoencoder", "tfrs"]:
        return (
            jsonify(
                {
                    "error": f'Invalid model name: {model_name}. Use "autoencoder" or "tfrs" for item recommendations.',
                    "status": "error",
                }
            ),
            400,
        )

    # Validate k
    if not isinstance(k, int) or k < 1 or k > 100:
        return (
            jsonify(
                {"error": "k must be an integer between 1 and 100", "status": "error"}
            ),
            400,
        )

    logger.info(f"Finding {k} similar items to '{item_name}' using {model_name}")

    # Get pipeline and find similar items
    pipeline = get_pipeline(model_name)
    similar_items = pipeline.get_similar_items(item_name=item_name, k=k)

    # Get full game details for each similar item
    similar_items_with_details = []
    for game_name in similar_items:
        for game in GAMES_DATA:
            if (
                game.get("title", "").lower() == game_name.lower()
                or game.get("app_name", "").lower() == game_name.lower()
            ):
                similar_items_with_details.append(game)
                break

    if not similar_items:
        logger.warning(f"No similar items found for '{item_name}'")
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f'No similar items found for "{item_name}". Item may not be in the catalog.',
                    "query_item": item_name,
                    "model_used": model_name,
                    "similar_items": [],
                    "similar_items_with_details": [],
                    "count": 0,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            200,
        )

    logger.info(f"Successfully found {len(similar_items)} similar items")

    return (
        jsonify(
            {
                "status": "success",
                "query_item": item_name,
                "model_used": model_name,
                "similar_items": similar_items,
                "similar_items_with_details": similar_items_with_details,
                "count": len(similar_items),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.route("/batch_recommend", methods=["POST"])
@validate_input(["model_name", "user_ids"])
@handle_errors
def batch_recommend():
    """
    Generate recommendations for multiple users in batch.

    Request JSON:
    {
        "model_name": "tfrs",
        "user_ids": ["user1", "user2", "user3"],
        "n_rec": 10  # optional, default 10
    }

    Response JSON:
    {
        "status": "success",
        "model_used": "tfrs",
        "results": {
            "user1": ["Game 1", "Game 2", ...],
            "user2": ["Game 1", "Game 2", ...],
            ...
        },
        "total_users": 3,
        "successful": 3,
        "failed": 0,
        "timestamp": "2024-01-01T00:00:00"
    }
    """
    data = request.get_json()

    model_name = data["model_name"].lower()
    user_ids = data["user_ids"]
    n_rec = data.get("n_rec", 10)

    # Validate model name
    if model_name not in ["tfrs", "mf"]:
        return (
            jsonify(
                {
                    "error": f'Invalid model name: {model_name}. Use "tfrs" or "mf" for user recommendations.',
                    "status": "error",
                }
            ),
            400,
        )

    # Validate user_ids
    if not isinstance(user_ids, list) or len(user_ids) == 0:
        return (
            jsonify({"error": "user_ids must be a non-empty list", "status": "error"}),
            400,
        )

    if len(user_ids) > 100:
        return (
            jsonify(
                {
                    "error": "Maximum 100 users allowed per batch request",
                    "status": "error",
                }
            ),
            400,
        )

    logger.info(f"Batch recommendation for {len(user_ids)} users using {model_name}")

    # Get pipeline
    pipeline = get_pipeline(model_name)

    # Generate recommendations for each user
    results = {}
    successful = 0
    failed = 0

    for user_id in user_ids:
        try:
            user_id_str = str(user_id)
            recommendations = pipeline.recommend(user_id=user_id_str, n_rec=n_rec)
            results[user_id_str] = recommendations
            if recommendations:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
            results[str(user_id)] = []
            failed += 1

    logger.info(f"Batch complete: {successful} successful, {failed} failed")

    return (
        jsonify(
            {
                "status": "success",
                "model_used": model_name,
                "results": results,
                "total_users": len(user_ids),
                "successful": successful,
                "failed": failed,
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.route("/model_info/<model_name>", methods=["GET"])
@handle_errors
def model_info(model_name):
    """
    Get information about a specific model.

    Response JSON:
    {
        "status": "success",
        "model_name": "tfrs",
        "is_loaded": true,
        "supports_user_recommendations": true,
        "supports_item_similarity": true,
        "description": "..."
    }
    """
    model_name = model_name.lower()

    model_descriptions = {
        "tfrs": {
            "name": "TensorFlow Recommenders (TFRS)",
            "description": "Two-tower neural architecture for efficient retrieval-based recommendations",
            "supports_user_recommendations": True,
            "supports_item_similarity": True,
            "best_for": "Fast, scalable recommendations with both collaborative and content-based features",
        },
        "mf": {
            "name": "Matrix Factorization",
            "description": "Classic collaborative filtering using user and item embeddings",
            "supports_user_recommendations": True,
            "supports_item_similarity": False,
            "best_for": "Personalized recommendations based on user-item interactions",
        },
        "autoencoder": {
            "name": "Autoencoder",
            "description": "Neural network-based content filtering using item features",
            "supports_user_recommendations": False,
            "supports_item_similarity": True,
            "best_for": "Finding similar items based on content features (genres, tags)",
        },
    }

    if model_name not in model_descriptions:
        return (
            jsonify(
                {
                    "error": f"Unknown model: {model_name}",
                    "status": "error",
                    "available_models": list(model_descriptions.keys()),
                }
            ),
            404,
        )

    info = model_descriptions[model_name]
    info["model_name"] = model_name
    info["is_loaded"] = model_name in MODEL_CACHE
    info["status"] = "success"

    return jsonify(info), 200


@app.route("/available_models", methods=["GET"])
def available_models():
    """
    List all available models and their status.

    Response JSON:
    {
        "status": "success",
        "models": [
            {
                "name": "tfrs",
                "loaded": true,
                "supports_user_recs": true,
                "supports_item_similarity": true
            },
            ...
        ]
    }
    """
    models = [
        {
            "name": "tfrs",
            "loaded": "tfrs" in MODEL_CACHE,
            "supports_user_recs": True,
            "supports_item_similarity": True,
        },
        {
            "name": "mf",
            "loaded": "mf" in MODEL_CACHE,
            "supports_user_recs": True,
            "supports_item_similarity": False,
        },
        {
            "name": "autoencoder",
            "loaded": "autoencoder" in MODEL_CACHE,
            "supports_user_recs": False,
            "supports_item_similarity": True,
        },
    ]

    return (
        jsonify(
            {
                "status": "success",
                "models": models,
                "total_loaded": len(MODEL_CACHE),
                "timestamp": datetime.now().isoformat(),
            }
        ),
        200,
    )


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return (
        jsonify(
            {
                "error": "Endpoint not found",
                "status": "error",
                "message": "The requested URL was not found on the server.",
            }
        ),
        404,
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return (
        jsonify(
            {
                "error": "Method not allowed",
                "status": "error",
                "message": "The method is not allowed for the requested URL.",
            }
        ),
        405,
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return (
        jsonify(
            {
                "error": "Internal server error",
                "status": "error",
                "message": "An internal server error occurred.",
            }
        ),
        500,
    )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Load data on startup
    logger.info("Starting data preprocessing...")
    try:
        load_games_data()
        load_user_ids()
        logger.info("Data preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        # Continue anyway, endpoints will handle missing data

    # Optional: Preload models on startup
    preload_models = os.environ.get("PRELOAD_MODELS", "false").lower() == "true"

    if preload_models:
        logger.info("Preloading models...")
        for model_name in ["tfrs", "mf", "autoencoder"]:
            try:
                get_pipeline(model_name)
            except Exception as e:
                logger.warning(f"Could not preload {model_name}: {e}")

    # Get configuration from environment variables
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting Flask application on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Loaded {len(GAMES_DATA)} games and {len(USER_IDS)} users")

    app.run(host=host, port=port, debug=debug)

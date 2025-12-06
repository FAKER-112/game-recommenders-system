import json
import os
import ast
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# --- Configuration & Data Setup ---
DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'steam_games.json')

# The sample data you provided, used to initialize the file if missing
SAMPLE_RAW_DATA = """
{u'publisher': u'Kotoshiro', u'genres': [u'Action', u'Casual', u'Indie', u'Simulation', u'Strategy'], u'app_name': u'Lost Summoner Kitty', u'title': u'Lost Summoner Kitty', u'url': u'http://store.steampowered.com/app/761140/Lost_Summoner_Kitty/', u'release_date': u'2018-01-04', u'tags': [u'Strategy', u'Action', u'Indie', u'Casual', u'Simulation'], u'discount_price': 4.49, u'reviews_url': u'http://steamcommunity.com/app/761140/reviews/?browsefilter=mostrecent&p=1', u'specs': [u'Single-player'], u'price': 4.99, u'early_access': False, u'id': u'761140', u'developer': u'Kotoshiro'}
{u'publisher': u'Making Fun, Inc.', u'genres': [u'Free to Play', u'Indie', u'RPG', u'Strategy'], u'app_name': u'Ironbound', u'sentiment': u'Mostly Positive', u'title': u'Ironbound', u'url': u'http://store.steampowered.com/app/643980/Ironbound/', u'release_date': u'2018-01-04', u'tags': [u'Free to Play', u'Strategy', u'Indie', u'RPG', u'Card Game', u'Trading Card Game', u'Turn-Based', u'Fantasy', u'Tactical', u'Dark Fantasy', u'Board Game', u'PvP', u'2D', u'Competitive', u'Replay Value', u'Character Customization', u'Female Protagonist', u'Difficult', u'Design & Illustration'], u'reviews_url': u'http://steamcommunity.com/app/643980/reviews/?browsefilter=mostrecent&p=1', u'specs': [u'Single-player', u'Multi-player', u'Online Multi-Player', u'Cross-Platform Multiplayer', u'Steam Achievements', u'Steam Trading Cards', u'In-App Purchases'], u'price': u'Free To Play', u'early_access': False, u'id': u'643980', u'developer': u'Secret Level SRL'}
{u'publisher': u'Poolians.com', u'genres': [u'Casual', u'Free to Play', u'Indie', u'Simulation', u'Sports'], u'app_name': u'Real Pool 3D - Poolians', u'sentiment': u'Mostly Positive', u'title': u'Real Pool 3D - Poolians', u'url': u'http://store.steampowered.com/app/670290/Real_Pool_3D__Poolians/', u'release_date': u'2017-07-24', u'tags': [u'Free to Play', u'Simulation', u'Sports', u'Casual', u'Indie', u'Multiplayer'], u'reviews_url': u'http://steamcommunity.com/app/670290/reviews/?browsefilter=mostrecent&p=1', u'specs': [u'Single-player', u'Multi-player', u'Online Multi-Player', u'In-App Purchases', u'Stats'], u'price': u'Free to Play', u'early_access': False, u'id': u'670290', u'developer': u'Poolians.com'}
{u'publisher': u'\u5f7c\u5cb8\u9886\u57df', u'genres': [u'Action', u'Adventure', u'Casual'], u'app_name': u'\u5f39\u70b8\u4eba2222', u'title': u'\u5f39\u70b8\u4eba2222', u'url': u'http://store.steampowered.com/app/767400/2222/', u'release_date': u'2017-12-07', u'tags': [u'Action', u'Adventure', u'Casual'], u'discount_price': 0.83, u'reviews_url': u'http://steamcommunity.com/app/767400/reviews/?browsefilter=mostrecent&p=1', u'specs': [u'Single-player'], u'price': 0.99, u'early_access': False, u'id': u'767400', u'developer': u'\u5f7c\u5cb8\u9886\u57df'}
{u'app_name': u'Log Challenge', u'tags': [u'Action', u'Indie', u'Casual', u'Sports'], u'url': u'http://store.steampowered.com/app/773570/Log_Challenge/', u'price': 2.99, u'discount_price': 1.79, u'reviews_url': u'http://steamcommunity.com/app/773570/reviews/?browsefilter=mostrecent&p=1', u'id': u'773570', u'early_access': False, u'specs': [u'Single-player', u'Full controller support', u'HTC Vive', u'Oculus Rift', u'Tracked Motion Controllers', u'Room-Scale']}
"""

def init_data():
    """Create data directory and sample file if they don't exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if not os.path.exists(DATA_FILE):
        print("Creating sample data file...")
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            f.write(SAMPLE_RAW_DATA.strip())

# --- Parsing Logic ---

def clean_game_data(game):
    """Clean and standardize game data."""
    # 1. Handle defaults
    defaults = {
        'publisher': 'Unknown',
        'genres': [],
        'app_name': 'Unknown Game',
        'title': '',
        'url': '#',
        'release_date': 'TBA',
        'tags': [],
        'discount_price': None,
        'reviews_url': '',
        'specs': [],
        'price': 0,
        'early_access': False,
        'id': f"gen_{os.urandom(4).hex()}", # Fallback ID
        'developer': 'Unknown',
        'sentiment': 'No reviews'
    }
    
    cleaned = {k: game.get(k, v) for k, v in defaults.items()}
    
    # 2. Fix Title
    if not cleaned['title'] and cleaned['app_name']:
        cleaned['title'] = cleaned['app_name']

    # 3. Handle Price (Convert 'Free to Play' strings to 0)
    raw_price = cleaned['price']
    if isinstance(raw_price, str):
        if 'free' in raw_price.lower():
            cleaned['price'] = 0.0
        else:
            try:
                cleaned['price'] = float(raw_price)
            except (ValueError, TypeError):
                cleaned['price'] = 0.0
    elif raw_price is None:
        cleaned['price'] = 0.0
    else:
        cleaned['price'] = float(raw_price)

    return cleaned

def load_games():
    """
    Load games from the customized Python-dictionary-string format.
    Uses ast.literal_eval which handles the u'string' format safely.
    """
    games = []
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                try:
                    # ast.literal_eval is the safest way to parse Python literal structures
                    # It handles {u'key': u'val', 'bool': False} automatically
                    game_raw = ast.literal_eval(line)
                    
                    # Clean and append
                    cleaned_game = clean_game_data(game_raw)
                    games.append(cleaned_game)
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Skipping malformed line: {line[:50]}... Error: {e}")
                    continue
                    
        return games
    except FileNotFoundError:
        return []

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/games')
def api_games():
    games = load_games()
    return jsonify(games)

@app.route('/api/stats')
def api_stats():
    games = load_games()
    if not games:
        return jsonify({
            'total': 0, 'free': 0, 'avg_price': 0, 'genres': 0
        })
        
    prices = [g['price'] for g in games if g['price'] > 0]
    all_genres = set()
    for g in games:
        all_genres.update(g.get('genres', []))
        
    return jsonify({
        'total': len(games),
        'free': sum(1 for g in games if g['price'] == 0),
        'avg_price': round(sum(prices) / len(prices), 2) if prices else 0,
        'genres': len(all_genres)
    })

if __name__ == '__main__':
    init_data()
    # Debug mode is on for development
    app.run(debug=True, port=5000)
"""
Debug script to test web application components
Run this to diagnose issues with the web app
"""

import os
import json
import requests
from pathlib import Path


def test_games_file():
    """Test if games file exists and is readable"""
    print("\n" + "="*70)
    print("  TESTING GAMES FILE")
    print("="*70)
    
    paths_to_check = [
        'data/steam_games.json',
        'data/raw/steam_games.json',
        '../data/steam_games.json'
    ]
    
    found = False
    for path in paths_to_check:
        print(f"\nChecking: {path}")
        if os.path.exists(path):
            print(f"  ✅ File exists")
            found = True
            
            # Test reading
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    print(f"  ✅ File readable")
                    print(f"  📊 Total lines: {len(lines)}")
                    
                    # Parse first few games
                    games_parsed = 0
                    for i, line in enumerate(lines[:10]):
                        try:
                            game = json.loads(line.strip())
                            games_parsed += 1
                            if i == 0:
                                print(f"\n  📝 Sample game:")
                                print(f"     Title: {game.get('title', 'N/A')}")
                                print(f"     ID: {game.get('id', 'N/A')}")
                                print(f"     Price: ${game.get('price', 0)}")
                                print(f"     Genres: {game.get('genres', [])}")
                        except json.JSONDecodeError as e:
                            print(f"  ⚠️  Line {i+1}: JSON parse error")
                    
                    print(f"\n  ✅ Successfully parsed {games_parsed}/10 sample games")
                    
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
            
            break
        else:
            print(f"  ❌ File not found")
    
    if not found:
        print(f"\n❌ Games file not found in any location!")
        print(f"\n💡 To fix:")
        print(f"   1. Make sure data/steam_games.json exists")
        print(f"   2. Or create sample data: python data_helper.py --create-sample")
    
    return found


def test_api_endpoint(base_url="http://localhost:5000"):
    """Test the /games API endpoint"""
    print("\n" + "="*70)
    print("  TESTING /games API ENDPOINT")
    print("="*70)
    
    try:
        print(f"\nTesting: {base_url}/games")
        response = requests.get(f"{base_url}/games?limit=5", timeout=5)
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ API is working")
            data = response.json()
            print(f"\n  📊 Response:")
            print(f"     Status: {data.get('status')}")
            print(f"     Games returned: {data.get('count')}")
            print(f"     Total games: {data.get('total')}")
            
            if data.get('games'):
                game = data['games'][0]
                print(f"\n  📝 First game:")
                print(f"     Title: {game.get('title')}")
                print(f"     ID: {game.get('id')}")
                print(f"     Price: ${game.get('price', 0)}")
            
            return True
        else:
            print(f"  ❌ API returned error")
            print(f"  Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to API")
        print(f"\n💡 Make sure Flask app is running:")
        print(f"   python app.py")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_recommendations_endpoint(base_url="http://localhost:5000"):
    """Test the recommendations endpoint"""
    print("\n" + "="*70)
    print("  TESTING /recommend_user API ENDPOINT")
    print("="*70)
    
    try:
        print(f"\nTesting: {base_url}/recommend_user")
        response = requests.post(
            f"{base_url}/recommend_user",
            json={
                "model_name": "tfrs",
                "user_id": "76561197970982479",
                "n_rec": 5
            },
            timeout=10
        )
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ Recommendations API is working")
            data = response.json()
            print(f"\n  📊 Response:")
            print(f"     Status: {data.get('status')}")
            print(f"     User: {data.get('user_id')}")
            print(f"     Count: {data.get('count')}")
            
            if data.get('recommendations'):
                print(f"\n  🎮 Recommended games:")
                for i, game in enumerate(data['recommendations'][:5], 1):
                    print(f"     {i}. {game}")
            
            return True
        else:
            print(f"  ❌ API returned error")
            data = response.json()
            print(f"  Error: {data.get('error')}")
            print(f"\n💡 Make sure:")
            print(f"   1. Models are trained")
            print(f"   2. Model artifacts exist in artifacts/ folder")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to API")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_browser_access(base_url="http://localhost:5000"):
    """Test if web interface is accessible"""
    print("\n" + "="*70)
    print("  TESTING WEB INTERFACE")
    print("="*70)
    
    try:
        print(f"\nTesting: {base_url}/")
        response = requests.get(base_url, timeout=5)
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ Web interface is accessible")
            
            # Check if it's HTML
            if 'text/html' in response.headers.get('Content-Type', ''):
                print(f"  ✅ Returns HTML content")
                
                # Check for key elements
                html = response.text
                checks = [
                    ('Steam Game Recommender', 'Page title'),
                    ('userSelect', 'User selector'),
                    ('recommendationsGrid', 'Recommendations grid'),
                    ('API_BASE_URL', 'JavaScript API config')
                ]
                
                print(f"\n  🔍 Checking HTML content:")
                for search_str, description in checks:
                    if search_str in html:
                        print(f"     ✅ Found {description}")
                    else:
                        print(f"     ⚠️  Missing {description}")
            else:
                print(f"  ⚠️  Not returning HTML (Content-Type: {response.headers.get('Content-Type')})")
            
            return True
        else:
            print(f"  ❌ Web interface returned error")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to web interface")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def print_browser_instructions():
    """Print instructions for testing in browser"""
    print("\n" + "="*70)
    print("  BROWSER TESTING INSTRUCTIONS")
    print("="*70)
    
    print("""
1. Open browser and navigate to: http://localhost:5000

2. Open browser console (F12 or Ctrl+Shift+I)

3. Check for errors in the Console tab

4. Look for these log messages:
   ✅ "Initializing application..."
   ✅ "Loading games data..."
   ✅ "Successfully loaded X unique games"
   ✅ "Loaded X sample users"

5. If you see errors:
   - Check Network tab for failed requests
   - Look at the error messages in console
   - Verify Flask app is running in terminal

6. Test the flow:
   - Select a user from dropdown
   - Should see "Loading recommendations..."
   - Should display game cards
   - Click a game to see details

Common Browser Console Errors:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"Failed to load games: HTTP 404"
→ Games file not found, check data/steam_games.json

"Failed: Failed to fetch recommendations"
→ Models not trained or API not responding

"Game not found in database: <name>"
→ Game names in recommendations don't match database
    """)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("  WEB APPLICATION DEBUG SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Games file
    results['games_file'] = test_games_file()
    
    # Test 2: API endpoints
    results['games_api'] = test_api_endpoint()
    results['recommendations_api'] = test_recommendations_endpoint()
    
    # Test 3: Web interface
    results['web_interface'] = test_browser_access()
    
    # Print browser instructions
    print_browser_instructions()
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All tests passed! Your web app should be working.")
        print("\n🌐 Open in browser: http://localhost:5000")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\n🔧 Common fixes:")
        if not results.get('games_file'):
            print("   - Ensure data/steam_games.json exists")
        if not results.get('games_api'):
            print("   - Make sure Flask app is running: python app.py")
        if not results.get('recommendations_api'):
            print("   - Train models: python -m src.pipeline.training_pipeline")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--games':
            test_games_file()
        elif sys.argv[1] == '--api':
            test_api_endpoint()
            test_recommendations_endpoint()
        elif sys.argv[1] == '--web':
            test_browser_access()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python debug_web.py [--games|--api|--web]")
    else:
        run_all_tests()
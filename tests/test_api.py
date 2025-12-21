"""
API Integration Test Suite for Game Recommendation System
===========================================================

This integration test script provides comprehensive end-to-end testing of the Flask API
defined in `app.py`. Unlike unit tests, this script makes actual HTTP requests to a running
Flask server to validate all API endpoints, data retrieval, recommendation logic, and error handling.

**IMPORTANT**: This script requires the Flask application to be running before execution.
Start the Flask app with `python app.py` before running these tests.

Test Coverage:
-------------

**Health & Info Endpoints (4 tests)**
- Health Check: `/health` - Verifies server status and model loading
- Home: `/` - Tests API information and statistics
- Available Models: `/available_models` - Lists all loaded models
- Model Info: `/model_info/<model_name>` - Gets details for specific models (tfrs, mf, autoencoder)

**Data Endpoints (3 test groups)**
- Userlist: `/api/userlist` - Retrieves list of all users
- Gamedata: `/api/gamedata` - Tests pagination and game search functionality
- Game Detail: `/api/game/<game_id>` - Tests game lookup by ID and title

**Recommendation Endpoints (3 test groups)**
- User Recommendations: `/recommend_user` - Tests user-based recommendations with validation
  * Valid model names (tfrs, mf)
  * Invalid model names
  * Missing user IDs
  * Invalid n_rec values (too large)

- Item Similarity: `/recommend_item` - Tests item-based similarity recommendations
  * Valid models (autoencoder, tfrs)
  * Invalid models (mf - doesn't support item similarity)
  * Missing item names

- Batch Recommendations: `/batch_recommend` - Tests batch processing for multiple users
  * Valid batch requests
  * Empty user lists
  * Too many users (limit validation)

**Error Handling Tests (3 tests)**
- 404 Not Found for nonexistent endpoints
- 405 Method Not Allowed for wrong HTTP methods
- 400 Bad Request for invalid JSON payloads

Testing Approach:
----------------
- **Integration Testing**: Makes real HTTP requests to live server
- **Status Code Validation**: Verifies correct HTTP status codes (200, 400, 404, 405)
- **Response Structure**: Validates JSON response format and required fields
- **Edge Cases**: Tests invalid inputs, missing parameters, and boundary conditions
- **Real Data**: Uses actual user IDs and game titles from the running application

Test Results:
------------
Tests report with color-coded PASS/FAIL indicators:
- ✓ PASS (green): Test succeeded with expected behavior
- ✗ FAIL (red): Test failed or unexpected behavior

Usage:
------
1. Start Flask application:
   ```bash
   python app.py
   ```

2. Run test suite (default localhost:5000):
   ```bash
   python tests/test_api.py
   ```

3. Run against different server:
   ```bash
   python tests/test_api.py http://localhost:8000
   ```

Test Output:
-----------
The script provides:
- Detailed test results with pass/fail status
- Response data summaries (counts, first results, etc.)
- Final test summary with pass rate
- Recommendations for failed tests

Example:
--------
```
Testing API at: http://localhost:5000
✓ PASS | Health Check
      Models loaded: ['tfrs', 'mf', 'autoencoder'], Games: 13047
✓ PASS | TFRS User Recommendations
      Got 5 recommendations, First: Counter-Strike: Global Offensive
...
TEST SUMMARY
Total Tests: 35
Passed: 35
Failed: 0
Pass Rate: 100.0%
```

Notes:
------
- This is NOT a unittest - it's an integration test script
- Requires Flask app to be running and fully initialized
- Tests depend on loaded models and data
- Some tests use real data from the application (user IDs, game titles)
- Default test user: "76561197970982479"
- Default test game: "Football Manager 2017"
"""

import requests
import json
from datetime import datetime


class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def print_header(self, text):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)

    def print_test(self, name, passed, details=""):
        """Print test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} | {name}")
        if details:
            print(f"      {details}")
        self.test_results.append((name, passed))

    def make_request(self, method, endpoint, data=None, params=None):
        """Make HTTP request and handle errors."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None

    def test_health_check(self):
        """Test health check endpoint."""
        self.print_header("Testing Health Check")

        response = self.make_request("GET", "/health")

        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Health Check",
                data.get("status") == "healthy",
                f"Models loaded: {data.get('models_loaded', [])}, Games: {data.get('counts', {}).get('games', 0)}",
            )
        else:
            self.print_test("Health Check", False, "Request failed")

    def test_home(self):
        """Test home endpoint."""
        self.print_header("Testing Home Endpoint")

        response = self.make_request("GET", "/")

        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Home Endpoint",
                data.get("status") == "success",
                f"Version: {data.get('version')}, Games: {data.get('stats', {}).get('total_games', 0)}",
            )
            endpoints = list(data.get("endpoints", {}).keys())
            print(f"      Available endpoints: {endpoints}")
        else:
            self.print_test("Home Endpoint", False, "Request failed")

    def test_userlist_endpoint(self):
        """Test userlist endpoint."""
        self.print_header("Testing Userlist Endpoint")

        response = self.make_request("GET", "/api/userlist")

        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Userlist Endpoint",
                data.get("status") == "success",
                f"Found {data.get('count', 0)} users",
            )

            users = data.get("users", [])
            if users:
                print(f"      First 3 users: {users[:3]}")
        else:
            self.print_test(
                "Userlist Endpoint",
                False,
                f"Status: {response.status_code if response else 'No response'}",
            )

    def test_gamedata_endpoint(self):
        """Test gamedata endpoint with pagination."""
        self.print_header("Testing Gamedata Endpoint")

        # Test 1: Basic pagination
        response = self.make_request(
            "GET", "/api/gamedata", params={"start": 0, "end": 5}
        )

        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Gamedata - Basic Pagination",
                data.get("status") == "success" and "games" in data,
                f"Got {len(data.get('games', []))} games, Total: {data.get('total_games', 0)}",
            )

            if data.get("games"):
                first_game = data["games"][0]
                print(f"      First game: {first_game.get('title', 'Unknown')}")

        # Test 2: With search
        response = self.make_request(
            "GET", "/api/gamedata", params={"start": 0, "end": 5, "search": "Counter"}
        )

        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Gamedata - With Search",
                data.get("status") == "success",
                f"Found {data.get('total_games', 0)} games matching 'Counter'",
            )
        else:
            self.print_test("Gamedata - With Search", False, "Request failed")

    def test_game_detail_endpoint(self):
        """Test game detail endpoint."""
        self.print_header("Testing Game Detail Endpoint")

        # First, get a game ID from the gamedata endpoint
        response = self.make_request(
            "GET", "/api/gamedata", params={"start": 0, "end": 1}
        )

        if response and response.status_code == 200:
            data = response.json()
            if data.get("games"):
                game = data["games"][0]
                game_id = game.get("id")
                game_title = game.get("title")

                # Test with ID
                response2 = self.make_request("GET", f"/api/game/{game_id}")

                if response2 and response2.status_code == 200:
                    data2 = response2.json()
                    self.print_test(
                        "Game Detail - By ID",
                        data2.get("status") == "success"
                        and data2.get("game", {}).get("id") == game_id,
                        f"Found game: {data2.get('game', {}).get('title', 'Unknown')}",
                    )

                # Test with title (if API supports it)
                response3 = self.make_request(
                    "GET", f'/api/game/{game_title.replace(" ", "%20")}'
                )

                if response3:
                    self.print_test(
                        "Game Detail - By Title",
                        response3.status_code
                        in [200, 404],  # Either found or not found
                        f"Status: {response3.status_code}",
                    )

                # Test with non-existent ID
                response4 = self.make_request("GET", "/api/game/nonexistent123")
                self.print_test(
                    "Game Detail - Non-existent ID",
                    response4.status_code == 404,
                    "Correctly returns 404 for non-existent game",
                )
            else:
                self.print_test(
                    "Game Detail Endpoint", False, "No games found to test with"
                )
        else:
            self.print_test("Game Detail Endpoint", False, "Could not fetch test game")

    def test_model_info(self):
        """Test model info endpoints."""
        self.print_header("Testing Model Info")

        for model_name in ["tfrs", "mf", "autoencoder"]:
            response = self.make_request("GET", f"/model_info/{model_name}")

            if response:
                if response.status_code == 200:
                    data = response.json()
                    self.print_test(
                        f"Model Info: {model_name}",
                        data.get("status") == "success",
                        f"Loaded: {data.get('is_loaded')}, Supports User Recs: {data.get('supports_user_recommendations')}",
                    )
                elif response.status_code == 404:
                    self.print_test(
                        f"Model Info: {model_name}", False, "Model not found"
                    )
                else:
                    self.print_test(
                        f"Model Info: {model_name}",
                        False,
                        f"Status: {response.status_code}",
                    )
            else:
                self.print_test(f"Model Info: {model_name}", False, "No response")

        # Test invalid model
        response = self.make_request("GET", "/model_info/invalid_model")
        self.print_test(
            "Model Info: Invalid Model",
            response.status_code == 404,
            "Correctly returns 404 for invalid model",
        )

    def test_available_models(self):
        """Test available models endpoint."""
        self.print_header("Testing Available Models")

        response = self.make_request("GET", "/available_models")

        if response and response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            self.print_test(
                "Available Models", len(models) == 3, f"Found {len(models)} models"
            )
        else:
            self.print_test("Available Models", False)

    def test_user_recommendations(self):
        """Test user recommendation endpoints."""
        self.print_header("Testing User Recommendations")

        # First get a real user ID from userlist
        response = self.make_request("GET", "/api/userlist")
        test_user_id = "76561197970982479"  # Default test user

        if response and response.status_code == 200:
            data = response.json()
            if data.get("users"):
                test_user_id = data["users"][0]

        test_cases = [
            {
                "name": "TFRS User Recommendations",
                "data": {"model_name": "tfrs", "user_id": test_user_id, "n_rec": 5},
                "expected_status": 200,
            },
            {
                "name": "MF User Recommendations",
                "data": {"model_name": "mf", "user_id": test_user_id, "n_rec": 5},
                "expected_status": 200,
            },
            {
                "name": "Invalid Model Name",
                "data": {"model_name": "invalid", "user_id": test_user_id, "n_rec": 5},
                "expected_status": 400,
            },
            {
                "name": "Missing User ID",
                "data": {"model_name": "tfrs", "n_rec": 5},
                "expected_status": 400,
            },
            {
                "name": "Invalid n_rec (too large)",
                "data": {"model_name": "tfrs", "user_id": test_user_id, "n_rec": 101},
                "expected_status": 400,
            },
        ]

        for test_case in test_cases:
            response = self.make_request("POST", "/recommend_user", test_case["data"])

            if response:
                passed = response.status_code == test_case["expected_status"]
                data = response.json() if response.status_code == 200 else {}

                if passed and response.status_code == 200:
                    details = f"Got {data.get('count', 0)} recommendations"
                    if data.get("recommendations"):
                        details += f", First: {data['recommendations'][0]}"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get("error", "")

                self.print_test(test_case["name"], passed, details)
            else:
                self.print_test(test_case["name"], False, "Request failed")

    def test_item_recommendations(self):
        """Test item similarity endpoints."""
        self.print_header("Testing Item Similarity")

        # First get a real game title from gamedata
        response = self.make_request(
            "GET", "/api/gamedata", params={"start": 0, "end": 1}
        )
        test_game_name = "Football Manager 2017"  # Default test game

        if response and response.status_code == 200:
            data = response.json()
            if data.get("games"):
                test_game_name = data["games"][0].get("title", test_game_name)

        test_cases = [
            {
                "name": "Autoencoder Similar Items",
                "data": {
                    "model_name": "autoencoder",
                    "item_name": test_game_name,
                    "k": 5,
                },
                "expected_status": 200,
            },
            {
                "name": "TFRS Similar Items",
                "data": {"model_name": "tfrs", "item_name": test_game_name, "k": 5},
                "expected_status": 200,
            },
            {
                "name": "Invalid Model for Item Similarity",
                "data": {"model_name": "mf", "item_name": test_game_name, "k": 5},
                "expected_status": 400,
            },
            {
                "name": "Missing Item Name",
                "data": {"model_name": "autoencoder", "k": 5},
                "expected_status": 400,
            },
        ]

        for test_case in test_cases:
            response = self.make_request("POST", "/recommend_item", test_case["data"])

            if response:
                passed = response.status_code == test_case["expected_status"]
                data = response.json() if response.status_code == 200 else {}

                if passed and response.status_code == 200:
                    details = f"Got {data.get('count', 0)} similar items"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get("error", "")

                self.print_test(test_case["name"], passed, details)
            else:
                self.print_test(test_case["name"], False, "Request failed")

    def test_batch_recommendations(self):
        """Test batch recommendation endpoint."""
        self.print_header("Testing Batch Recommendations")

        # Get real user IDs
        response = self.make_request("GET", "/api/userlist")
        user_ids = ["76561197970982479", "76561198001103300"]  # Default test users

        if response and response.status_code == 200:
            data = response.json()
            if data.get("users") and len(data["users"]) >= 2:
                user_ids = data["users"][:2]

        test_cases = [
            {
                "name": "Batch TFRS Recommendations",
                "data": {"model_name": "tfrs", "user_ids": user_ids, "n_rec": 3},
                "expected_status": 200,
            },
            {
                "name": "Empty User List",
                "data": {"model_name": "tfrs", "user_ids": [], "n_rec": 5},
                "expected_status": 400,
            },
            {
                "name": "Too Many Users",
                "data": {
                    "model_name": "tfrs",
                    "user_ids": [f"user_{i}" for i in range(101)],
                    "n_rec": 5,
                },
                "expected_status": 400,
            },
        ]

        for test_case in test_cases:
            response = self.make_request("POST", "/batch_recommend", test_case["data"])

            if response:
                passed = response.status_code == test_case["expected_status"]
                data = response.json() if response.status_code == 200 else {}

                if passed and response.status_code == 200:
                    details = f"Successful: {data.get('successful', 0)}/{data.get('total_users', 0)}"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get("error", "")

                self.print_test(test_case["name"], passed, details)
            else:
                self.print_test(test_case["name"], False, "Request failed")

    def test_error_handling(self):
        """Test error handling."""
        self.print_header("Testing Error Handling")

        # Test 404
        response = self.make_request("GET", "/nonexistent_endpoint")
        self.print_test(
            "404 Not Found", response.status_code == 404, "Correctly returns 404"
        )

        # Test 405 Method Not Allowed
        response = self.make_request("GET", "/recommend_user")
        self.print_test(
            "405 Method Not Allowed",
            response.status_code == 405,
            "Correctly returns 405",
        )

        # Test invalid JSON
        try:
            url = f"{self.base_url}/recommend_user"
            response = self.session.post(
                url, data="invalid json", headers={"Content-Type": "application/json"}
            )
            self.print_test(
                "Invalid JSON Handling",
                response.status_code == 400,
                "Correctly handles malformed JSON",
            )
        except Exception as e:
            self.print_test("Invalid JSON Handling", False, str(e))

    def test_new_data_endpoints(self):
        """Test all new data-related endpoints."""
        self.print_header("Testing New Data Endpoints")

        # Test userlist
        self.test_userlist_endpoint()

        # Test gamedata
        self.test_gamedata_endpoint()

        # Test game detail
        self.test_game_detail_endpoint()

    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "=" * 70)
        print("  GAME RECOMMENDATION API - TEST SUITE (UPDATED)")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("  Testing new app.py with data endpoints")
        print("=" * 70)

        try:
            # Run all test groups
            self.test_health_check()
            self.test_home()
            self.test_new_data_endpoints()
            self.test_available_models()
            self.test_model_info()
            self.test_user_recommendations()
            self.test_item_recommendations()
            self.test_batch_recommendations()
            self.test_error_handling()

            # Print summary
            self.print_summary()

        except KeyboardInterrupt:
            print("\n\nTests interrupted by user")
            self.print_summary()
        except Exception as e:
            print(f"\n\nUnexpected error: {e}")
            import traceback

            traceback.print_exc()

    def print_summary(self):
        """Print test summary."""
        self.print_header("TEST SUMMARY")

        total = len(self.test_results)
        passed = sum(1 for _, result in self.test_results if result)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"\033[92mPassed: {passed}\033[0m")
        print(f"\033[91mFailed: {failed}\033[0m")
        print(f"Pass Rate: {pass_rate:.1f}%")

        if failed > 0:
            print("\n\033[91mFailed Tests:\033[0m")
            for name, result in self.test_results:
                if not result:
                    print(f"  - {name}")

        print("\n" + "=" * 70)

        # Additional recommendations
        if pass_rate < 100:
            print("\n\033[93mRecommendations:\033[0m")
            print("1. Make sure app.py is running with data loaded")
            print("2. Check if models are trained and available")
            print("3. Verify data files exist in data/raw/ and data/processed/")
            print("4. Check Flask logs for any startup errors")

        print("\n")


if __name__ == "__main__":
    import sys

    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

    print(f"Testing API at: {base_url}")
    print("Make sure the Flask app is running before executing tests!\n")
    print("Note: This tester now includes tests for the new data endpoints:")
    print("  - /api/userlist")
    print("  - /api/gamedata (with pagination)")
    print("  - /api/game/<game_id>")
    print("\nThe app.py must be started with the new modifications.")

    tester = APITester(base_url)
    tester.run_all_tests()

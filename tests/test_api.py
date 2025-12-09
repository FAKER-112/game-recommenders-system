"""
API Testing Script for Game Recommendation System
Run this script to test all API endpoints
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
        
    def make_request(self, method, endpoint, data=None):
        """Make HTTP request and handle errors."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
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
        
        response = self.make_request('GET', '/health')
        
        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Health Check",
                data.get('status') == 'healthy',
                f"Models loaded: {data.get('models_loaded', [])}"
            )
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            self.print_test("Health Check", False, "Request failed")
    
    def test_home(self):
        """Test home endpoint."""
        self.print_header("Testing Home Endpoint")
        
        response = self.make_request('GET', '/')
        
        if response and response.status_code == 200:
            data = response.json()
            self.print_test(
                "Home Endpoint",
                data.get('status') == 'success',
                f"Version: {data.get('version')}"
            )
            print(f"Available endpoints: {list(data.get('endpoints', {}).keys())}")
        else:
            self.print_test("Home Endpoint", False, "Request failed")
    
    def test_model_info(self):
        """Test model info endpoints."""
        self.print_header("Testing Model Info")
        
        for model_name in [ 'mf', 'autoencoder']:
            response = self.make_request('GET', f'/model_info/{model_name}')
            
            if response and response.status_code == 200:
                data = response.json()
                self.print_test(
                    f"Model Info: {model_name}",
                    data.get('status') == 'success',
                    f"Loaded: {data.get('is_loaded')}, User Recs: {data.get('supports_user_recommendations')}"
                )
            else:
                self.print_test(f"Model Info: {model_name}", False)
        
        # Test invalid model
        response = self.make_request('GET', '/model_info/invalid_model')
        self.print_test(
            "Model Info: Invalid Model",
            response.status_code == 404,
            "Correctly returns 404 for invalid model"
        )
    
    def test_available_models(self):
        """Test available models endpoint."""
        self.print_header("Testing Available Models")
        
        response = self.make_request('GET', '/available_models')
        
        if response and response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            self.print_test(
                "Available Models",
                len(models) == 3,
                f"Found {len(models)} models"
            )
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            self.print_test("Available Models", False)
    
    def test_user_recommendations(self):
        """Test user recommendation endpoints."""
        self.print_header("Testing User Recommendations")
        
        test_cases = [
            {
                "name": "TFRS User Recommendations",
                "data": {
                    "model_name": "tfrs",
                    "user_id": "76561197970982479",
                    "n_rec": 5
                },
                "expected_status": 200
            },
            {
                "name": "MF User Recommendations",
                "data": {
                    "model_name": "mf",
                    "user_id": "76561197970982479",
                    "n_rec": 5
                },
                "expected_status": 200
            },
            {
                "name": "Invalid Model Name",
                "data": {
                    "model_name": "invalid",
                    "user_id": "76561197970982479",
                    "n_rec": 5
                },
                "expected_status": 400
            },
            {
                "name": "Missing User ID",
                "data": {
                    "model_name": "tfrs",
                    "n_rec": 5
                },
                "expected_status": 400
            },
            {
                "name": "Invalid n_rec (too large)",
                "data": {
                    "model_name": "tfrs",
                    "user_id": "76561197970982479",
                    "n_rec": 101
                },
                "expected_status": 400
            }
        ]
        
        for test_case in test_cases:
            response = self.make_request('POST', '/recommend_user', test_case['data'])
            
            if response:
                passed = response.status_code == test_case['expected_status']
                data = response.json() if response.status_code == 200 else {}
                
                if passed and response.status_code == 200:
                    details = f"Got {data.get('count', 0)} recommendations"
                    if data.get('recommendations'):
                        details += f", First: {data['recommendations'][0]}"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get('error', '')
                
                self.print_test(test_case['name'], passed, details)
                
                if passed and response.status_code == 200 and data.get('recommendations'):
                    print(f"      Sample recommendations: {data['recommendations'][:3]}")
            else:
                self.print_test(test_case['name'], False, "Request failed")
    
    def test_item_recommendations(self):
        """Test item similarity endpoints."""
        self.print_header("Testing Item Similarity")
        
        test_cases = [
            {
                "name": "Autoencoder Similar Items",
                "data": {
                    "model_name": "autoencoder",
                    "item_name": "Football Manager 2017",
                    "k": 5
                },
                "expected_status": 200
            },
            {
                "name": "TFRS Similar Items",
                "data": {
                    "model_name": "tfrs",
                    "item_name": "Football Manager 2017",
                    "k": 5
                },
                "expected_status": 200
            },
            {
                "name": "Invalid Model for Item Similarity",
                "data": {
                    "model_name": "mf",
                    "item_name": "Football Manager 2017",
                    "k": 5
                },
                "expected_status": 400
            },
            {
                "name": "Missing Item Name",
                "data": {
                    "model_name": "autoencoder",
                    "k": 5
                },
                "expected_status": 400
            },
            {
                "name": "Non-existent Item",
                "data": {
                    "model_name": "autoencoder",
                    "item_name": "NonExistentGame12345",
                    "k": 5
                },
                "expected_status": 200  # Should return empty list with success
            }
        ]
        
        for test_case in test_cases:
            response = self.make_request('POST', '/recommend_item', test_case['data'])
            
            if response:
                passed = response.status_code == test_case['expected_status']
                data = response.json() if response.status_code == 200 else {}
                
                if passed and response.status_code == 200:
                    details = f"Got {data.get('count', 0)} similar items"
                    if data.get('similar_items'):
                        details += f", First: {data['similar_items'][0]}"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get('error', '')
                
                self.print_test(test_case['name'], passed, details)
                
                if passed and response.status_code == 200 and data.get('similar_items'):
                    print(f"      Similar items: {data['similar_items'][:3]}")
            else:
                self.print_test(test_case['name'], False, "Request failed")
    
    def test_batch_recommendations(self):
        """Test batch recommendation endpoint."""
        self.print_header("Testing Batch Recommendations")
        
        test_cases = [
            {
                "name": "Batch TFRS Recommendations",
                "data": {
                    "model_name": "tfrs",
                    "user_ids": ["76561197970982479", "76561198001103300"],
                    "n_rec": 3
                },
                "expected_status": 200
            },
            {
                "name": "Empty User List",
                "data": {
                    "model_name": "tfrs",
                    "user_ids": [],
                    "n_rec": 5
                },
                "expected_status": 400
            },
            {
                "name": "Too Many Users",
                "data": {
                    "model_name": "tfrs",
                    "user_ids": [f"user_{i}" for i in range(101)],
                    "n_rec": 5
                },
                "expected_status": 400
            }
        ]
        
        for test_case in test_cases:
            response = self.make_request('POST', '/batch_recommend', test_case['data'])
            
            if response:
                passed = response.status_code == test_case['expected_status']
                data = response.json() if response.status_code == 200 else {}
                
                if passed and response.status_code == 200:
                    details = f"Successful: {data.get('successful', 0)}/{data.get('total_users', 0)}"
                elif not passed:
                    details = f"Expected {test_case['expected_status']}, got {response.status_code}"
                else:
                    details = response.json().get('error', '')
                
                self.print_test(test_case['name'], passed, details)
            else:
                self.print_test(test_case['name'], False, "Request failed")
    
    def test_error_handling(self):
        """Test error handling."""
        self.print_header("Testing Error Handling")
        
        # Test 404
        response = self.make_request('GET', '/nonexistent_endpoint')
        self.print_test(
            "404 Not Found",
            response.status_code == 404,
            "Correctly returns 404"
        )
        
        # Test 405 Method Not Allowed
        response = self.make_request('GET', '/recommend_user')
        self.print_test(
            "405 Method Not Allowed",
            response.status_code == 405,
            "Correctly returns 405"
        )
        
        # Test invalid JSON
        try:
            url = f"{self.base_url}/recommend_user"
            response = self.session.post(
                url,
                data="invalid json",
                headers={'Content-Type': 'application/json'}
            )
            self.print_test(
                "Invalid JSON Handling",
                response.status_code == 400,
                "Correctly handles malformed JSON"
            )
        except Exception as e:
            self.print_test("Invalid JSON Handling", False, str(e))
    
    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "=" * 70)
        print("  GAME RECOMMENDATION API - TEST SUITE")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70)
        
        try:
            # Run all test groups
            self.test_health_check()
            self.test_home()
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
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    print(f"Testing API at: {base_url}")
    print("Make sure the Flask app is running before executing tests!\n")
    
    tester = APITester(base_url)
    tester.run_all_tests()
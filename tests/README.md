# Test Suite Documentation

This directory contains comprehensive test coverage for the Game Recommendation System. The test suite includes unit tests for data processing and models, as well as integration tests for the Flask API.

## Overview

The test suite is organized into five main test modules:
1. **test_data.py** - Data processing pipeline tests
2. **test_model.py** - Model building, training, and evaluation tests
3. **test_predict_pipeline.py** - Prediction pipeline tests
4. **test_train_pipeline.py** - Training pipeline tests
5. **test_api.py** - API integration tests (requires running server)

## Test Coverage Summary

| Module | Test Classes | Total Tests | Coverage |
|--------|-------------|-------------|----------|
| test_data.py | 3 | 17 tests | Data ingestion, cleaning, feature engineering |
| test_model.py | 3 | 16 tests | Model building, training, evaluation |
| test_predict_pipeline.py | 1 | 10+ tests | Inference and recommendations |
| test_train_pipeline.py | 1 | 5+ tests | End-to-end training workflow |
| test_api.py | 1 | 35+ tests | All API endpoints and error handling |

## Test Modules

### 1. test_data.py

**Purpose**: Unit tests for data processing modules

**Test Classes**:
- `TestLoadDataService` (6 tests) - Data ingestion and download
- `TestCleanDataService` (5 tests) - Data cleaning and merging  
- `TestFeatureEngineeringService` (6 tests) - Feature creation and splitting

**Coverage**:
- ✅ Configuration loading and validation
- ✅ File download with skip logic for existing files
- ✅ User-item data processing and flattening
- ✅ Steam games data parsing and filtering
- ✅ Data merging with left joins
- ✅ Rating creation from playtime (log transformation)
- ✅ Genres/tags parsing (multiple formats)
- ✅ Item text generation
- ✅ Train/test splitting (80/20)
- ✅ Exception handling and error cases

**Mocking Strategy**:
- All file I/O operations mocked
- Network downloads mocked
- Config file loading mocked
- No actual data files required

**Usage**:
```bash
# Run all data tests
python -m pytest tests/test_data.py -v

# Run specific test class
python -m pytest tests/test_data.py::TestLoadDataService -v

# Run with unittest
python -m unittest tests.test_data
```

---

### 2. test_model.py

**Purpose**: Unit tests for model building, training, and evaluation

**Test Classes**:
- `TestModelBuilder` (7 tests) - Model architectures and data preparation
- `TestModelTrainingService` (5 tests) - Training workflows and MLflow integration
- `TestModelEvaluationService` (4 tests) - Evaluation metrics and routing

**Coverage**:
- ✅ Autoencoder data preparation (TF-IDF vectorization)
- ✅ Matrix Factorization data preparation (label encoding)
- ✅ TFRS data preparation (TensorFlow Datasets, vocabularies)
- ✅ Model architecture building for all three models
- ✅ Training workflows with MLflow logging
- ✅ Model saving (H5 and SavedModel formats)
- ✅ Context artifact persistence (encoders, vocabularies)
- ✅ Evaluation metric calculation (Precision@K, MAP@K, NDCG@K)
- ✅ Method routing for different model types
- ✅ Exception handling during training and evaluation

**Mocking Strategy**:
- MLflow operations fully mocked
- File operations mocked (pickle, JSON, CSV)
- Model training mocked (no actual training)
- TensorFlow operations mocked where needed

**Usage**:
```bash
# Run all model tests
python -m pytest tests/test_model.py -v

# Run specific test class
python -m pytest tests/test_model.py::TestModelBuilder -v

# Run with coverage
python -m pytest tests/test_model.py --cov=src/models
```

---

### 3. test_predict_pipeline.py

**Purpose**: Tests for prediction pipeline and inference engine

**Coverage**:
- ✅ Pipeline initialization and model loading
- ✅ User recommendation generation
- ✅ Item similarity search
- ✅ Context artifact loading (encoders, vocabularies)
- ✅ ID to name mapping
- ✅ Error handling for missing models/data

**Usage**:
```bash
# Run prediction pipeline tests
python -m pytest tests/test_predict_pipeline.py -v
```

---

### 4. test_train_pipeline.py

**Purpose**: Tests for end-to-end training pipeline orchestration

**Coverage**:
- ✅ Complete pipeline execution
- ✅ Data ingestion → cleaning → feature engineering → training
- ✅ Configuration management
- ✅ Directory creation and artifact saving
- ✅ Pipeline routing for different models

**Usage**:
```bash
# Run training pipeline tests
python -m pytest tests/test_train_pipeline.py -v
```

---

### 5. test_api.py ⚠️

**Purpose**: Integration tests for Flask API (requires running server)

**Important**: This is an **integration test**, not a unit test. The Flask application must be running before executing these tests.

**Test Coverage**:

**Health & Info Endpoints** (4 tests):
- `/health` - Server health check
- `/` - API information and statistics
- `/available_models` - List loaded models
- `/model_info/<model_name>` - Model details

**Data Endpoints** (3 test groups):
- `/api/userlist` - User list retrieval
- `/api/gamedata` - Game data with pagination and search
- `/api/game/<game_id>` - Game detail lookup

**Recommendation Endpoints** (3 test groups):
- `/recommend_user` - User-based recommendations
  - Valid models (TFRS, MF)
  - Invalid model names
  - Missing parameters
  - Invalid n_rec values
- `/recommend_item` - Item similarity
  - Valid models (Autoencoder, TFRS)
  - Invalid models
  - Missing item names
- `/batch_recommend` - Batch user recommendations
  - Valid batch requests
  - Empty user lists
  - Too many users (limit validation)

**Error Handling Tests** (3 tests):
- 404 Not Found
- 405 Method Not Allowed
- 400 Bad Request (malformed JSON)

**Usage**:
```bash
# 1. Start Flask application first
python app.py

# 2. In another terminal, run API tests
python tests/test_api.py

# 3. Or test against different server
python tests/test_api.py http://localhost:8000
```

**Output Example**:
```
======================================================================
  GAME RECOMMENDATION API - TEST SUITE (UPDATED)
  2024-12-14 18:52:40
  Testing new app.py with data endpoints
======================================================================

✓ PASS | Health Check
      Models loaded: ['tfrs', 'mf', 'autoencoder'], Games: 13047
✓ PASS | TFRS User Recommendations
      Got 5 recommendations, First: Counter-Strike: Global Offensive
✓ PASS | Gamedata - Basic Pagination
      Got 5 games, Total: 13047

======================================================================
  TEST SUMMARY
======================================================================

Total Tests: 35
Passed: 35
Failed: 0
Pass Rate: 100.0%
```

## Running All Tests

### Using pytest (Recommended)

```bash
# Run all tests (excluding integration tests)
python -m pytest tests/ -v --ignore=tests/test_api.py

# Run all tests including integration tests (requires running app)
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_data.py -v

# Run specific test class
python -m pytest tests/test_data.py::TestLoadDataService -v

# Run specific test method
python -m pytest tests/test_data.py::TestLoadDataService::test_init_success -v
```

### Using unittest

```bash
# Discover and run all tests
python -m unittest discover tests/

# Run specific test module
python -m unittest tests.test_data

# Run specific test class
python -m unittest tests.test_data.TestLoadDataService

# Run with verbose output
python -m unittest discover tests/ -v
```

## Testing Patterns and Best Practices

### 1. Mocking Strategy

All unit tests use extensive mocking to ensure:
- **Fast execution** (no actual file I/O or training)
- **Isolation** from external dependencies
- **Reproducibility** (consistent results)

**Commonly Mocked Components**:
```python
from unittest.mock import patch, MagicMock, mock_open

@patch("src.data.load_data.load_config")
@patch("os.makedirs")
@patch("pandas.read_csv")
def test_example(mock_read_csv, mock_makedirs, mock_load_config):
    # Test implementation
    pass
```

### 2. Test Data

Tests use minimal synthetic data:
```python
# Example test data
mock_df = pd.DataFrame({
    "user_id": ["user1", "user2"],
    "item_id": ["123", "456"],
    "rating": [1.0, 1.0]
})
```

### 3. Assertion Patterns

**Type checking**:
```python
self.assertIsInstance(model, tf.keras.Model)
```

**Value validation**:
```python
self.assertEqual(len(recommendations), 10)
self.assertGreater(metric_value, 0)
```

**Exception testing**:
```python
with self.assertRaises(CustomException):
    service.run()
```

### 4. MLflow Mocking

```python
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
def test_training(mock_log_metric, mock_log_params, mock_start_run):
    # Mock MLflow run context
    mock_run = MagicMock()
    mock_start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_start_run.return_value.__exit__ = MagicMock(return_value=False)
    
    # Run test
    service.train_model()
    
    # Verify logging
    mock_log_params.assert_called_once()
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --ignore=tests/test_api.py --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Coverage Goals

Target coverage by module:
- **Data Processing**: 90%+ coverage
- **Models**: 80%+ coverage (excluding TFRS internals)
- **Pipelines**: 85%+ coverage
- **API**: 90%+ coverage (via integration tests)

## Writing New Tests

### Template for Unit Tests

```python
import unittest
from unittest.mock import patch, MagicMock

class TestNewFeature(unittest.TestCase):
    """Test suite for new feature"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = {...}
    
    @patch("module.dependency")
    def test_functionality(self, mock_dependency):
        """Test description"""
        # Arrange
        mock_dependency.return_value = expected_value
        
        # Act
        result = function_to_test()
        
        # Assert
        self.assertEqual(result, expected_value)
        mock_dependency.assert_called_once()
    
    def tearDown(self):
        """Clean up after tests"""
        pass

if __name__ == "__main__":
    unittest.main()
```

### Template for Integration Tests

```python
def test_api_endpoint(self):
    """Test specific API endpoint"""
    # Arrange
    test_data = {"key": "value"}
    
    # Act
    response = self.make_request("POST", "/endpoint", test_data)
    
    # Assert
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["status"], "success")
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure project root is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Test Failures Due to Missing Mocks**
```python
# Make sure all external dependencies are mocked
@patch("src.module.external_dependency")
def test_function(mock_external):
    pass
```

**3. Integration Test Failures**
```bash
# Verify Flask app is running
curl http://localhost:5000/health

# Check if models are loaded
python -c "import os; print(os.listdir('artifacts/models'))"
```

**4. TensorFlow Warnings**
```python
# Suppress TensorFlow warnings in tests
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```

## Test Maintenance

**Guidelines**:
1. Update tests when changing APIs or interfaces
2. Add tests for all new features
3. Keep test data minimal but representative
4. Use descriptive test names
5. Document complex test scenarios
6. Maintain fast execution time (<5 seconds for unit tests)
7. Mock all I/O and external dependencies

## Dependencies

**Required**:
- `pytest` - Test framework
- `unittest` - Python standard library
- `unittest.mock` - Mocking utilities

**Optional**:
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel test execution
- `requests` - For API integration tests

**Installation**:
```bash
pip install pytest pytest-cov pytest-xdist requests
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Python mocking guide](https://docs.python.org/3/library/unittest.mock.html)
- [TensorFlow testing guide](https://www.tensorflow.org/guide/test)

---

**Test Suite Version**: 1.0  
**Last Updated**: 2024-12-14  
**Maintainer**: Development Team

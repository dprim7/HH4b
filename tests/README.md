# HH4b Test Suite

Comprehensive test suite following 2025 best practices for software engineering.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── pytest.ini               # Pytest settings
├── test_package.py          # Basic package tests
├── test_hh_vars.py          # Configuration variable tests
├── test_basic.py            # Core functionality tests (no heavy dependencies)
├── test_utils.py            # Utils module tests (requires numpy, pandas)
├── test_run_utils.py        # Run utilities tests
├── test_integration.py      # Integration tests
└── README.md                # This file
```

## Running Tests

### Quick Tests (No Dependencies)
Run basic tests that don't require scientific stack:
```bash
pytest tests/test_package.py tests/test_hh_vars.py tests/test_basic.py -v
```

### All Tests
Run all tests (requires full dependencies):
```bash
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ --cov=HH4b --cov-report=html
```

### Specific Test Categories
```bash
# Only unit tests
pytest -m unit

# Only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Only tests that don't require data
pytest -m "not requires_data"
```

## Test Categories

Tests are marked with the following categories:

- `unit`: Fast, isolated unit tests
- `integration`: Integration tests that test component interactions
- `slow`: Tests that take significant time
- `requires_data`: Tests that need actual data files
- `requires_external`: Tests that need external tools (git, condor, etc.)

## Writing Tests

### Best Practices

1. **Use Fixtures**: Use pytest fixtures from `conftest.py` for test data
2. **Parametrize**: Use `@pytest.mark.parametrize` for testing multiple scenarios
3. **Mark Tests**: Add appropriate markers (`@pytest.mark.unit`, etc.)
4. **Mock External Resources**: Use mocks for external APIs, file systems, etc.
5. **Test Edge Cases**: Include tests for error conditions and boundary cases
6. **Clear Names**: Use descriptive test names that explain what is being tested

### Example Test

```python
import pytest
from HH4b.utils import check_selector

class TestMyFeature:
    \"\"\"Tests for my feature.\"\"\"
    
    @pytest.mark.unit
    def test_basic_functionality(self):
        \"\"\"Test basic functionality works as expected.\"\"\"
        result = check_selector("QCD_HT100to200", "QCD?")
        assert result is True
    
    @pytest.mark.parametrize("sample,pattern,expected", [
        ("QCD_HT100to200", "QCD?", True),
        ("TTbar", "TT?", True),
        ("WJets", "QCD?", False),
    ])
    def test_multiple_scenarios(self, sample, pattern, expected):
        \"\"\"Test multiple scenarios with parametrize.\"\"\"
        result = check_selector(sample, pattern)
        assert result == expected
```

## Fixtures

Common fixtures available in `conftest.py`:

- `temp_dir`: Temporary directory that's cleaned up after test
- `mock_pickle_file`: Single pickle file with test data
- `mock_pickles_dir`: Directory with multiple pickle files
- `sample_year_configs`: Year and luminosity configurations
- `mock_xsecs`: Cross-section data

## CI/CD Integration

Tests can be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pytest tests/ -v --junitxml=test-results.xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Missing Dependencies

If you get import errors, install the full requirements:
```bash
pip install -r requirements.txt
```

For development, also install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist pytest-mock
```

### Test Failures

1. Check that you're in the correct directory
2. Ensure the package is installed: `pip install -e .`
3. Check for conflicting Python versions
4. Clear pytest cache: `pytest --cache-clear`

## Test Coverage

Target test coverage goals:
- **Overall**: >80%
- **Core utilities**: >90%
- **Critical paths**: 100%

View coverage report:
```bash
pytest --cov=HH4b --cov-report=html
open htmlcov/index.html
```

## Future Improvements

- [ ] Add performance benchmarking tests
- [ ] Add property-based testing with Hypothesis
- [ ] Add mutation testing with mutmut
- [ ] Expand integration tests for full workflows
- [ ] Add tests for visualization functions
- [ ] Add tests for machine learning components

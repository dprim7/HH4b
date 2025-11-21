"""
Pytest configuration and shared fixtures for HH4b tests.

Following 2025 best practices:
- Comprehensive fixtures for test isolation
- Proper resource cleanup
- Parametrized test support
- Mock data generation
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pickle_file(temp_dir):
    """Create a mock pickle file with test data."""
    data = {
        "2022": {
            "test_sample": {
                "nevents": 1000,
                "cutflow": {"initial": 1000, "after_cuts": 500},
            }
        }
    }
    pickle_path = temp_dir / "test.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(data, f)
    return pickle_path


@pytest.fixture
def mock_pickles_dir(temp_dir):
    """Create a directory with multiple pickle files."""
    pickles_dir = temp_dir / "pickles"
    pickles_dir.mkdir()

    for i in range(3):
        data = {
            "2022": {
                "test_sample": {
                    "nevents": 100 * (i + 1),
                    "cutflow": {"initial": 100 * (i + 1), "after_cuts": 50 * (i + 1)},
                }
            }
        }
        pickle_path = pickles_dir / f"test_{i}.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(data, f)

    return pickles_dir


@pytest.fixture
def sample_year_configs():
    """Provide sample year configurations for testing."""
    return {
        "years": ["2022", "2022EE", "2023"],
        "lumi": {"2022": 7980.5, "2022EE": 26671.6, "2023": 18084.4},
    }


@pytest.fixture
def mock_xsecs():
    """Provide mock cross-section data."""
    return {
        "QCD_HT100to200": {"xsec": 23700000.0, "kfactor": 1.0, "br": 1.0},
        "TTto4Q": {"xsec": 377.96, "kfactor": 1.0, "br": 1.0},
        "GluGlutoHHto4B": {"xsec": 0.03105, "kfactor": 1.0, "br": 0.5824**2},
    }


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests to prevent interference."""
    import logging

    # Store original handlers
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    yield

    # Restore original state
    root_logger.handlers = original_handlers
    root_logger.level = original_level


# Parametrization helpers for common test scenarios
YEARS = ["2022", "2022EE", "2023", "2023BPix"]
SAMPLE_TYPES = ["signal", "background", "data"]
PROCESSORS = ["skimmer", "ttSkimmer"]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_data: Tests that require actual data files")
    config.addinivalue_line(
        "markers", "requires_external: Tests that require external dependencies"
    )

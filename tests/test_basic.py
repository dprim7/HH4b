"""
Basic unit tests for core HH4b functionality that don't require heavy dependencies.

These tests can run quickly without full scientific stack.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest


class TestBasicImports:
    """Test that core modules can be imported."""

    def test_import_hh4b(self):
        """Test basic package import."""
        import HH4b

        assert HH4b.__version__

    def test_import_hh_vars(self):
        """Test importing hh_vars module."""
        from HH4b import hh_vars

        assert hasattr(hh_vars, "years")
        assert hasattr(hh_vars, "LUMI")
        assert hasattr(hh_vars, "DATA_SAMPLES")


class TestCheckSelector:
    """Test check_selector function without dependencies."""

    def test_exact_match(self):
        """Test exact sample name matching."""
        from HH4b.utils import check_selector

        assert check_selector("QCD_HT100to200", "QCD_HT100to200") is True
        assert check_selector("QCD_HT100to200", "QCD_HT200to300") is False

    def test_wildcard_match(self):
        """Test wildcard matching with ? suffix."""
        from HH4b.utils import check_selector

        assert check_selector("QCD_HT100to200", "QCD_HT?") is True
        assert check_selector("QCD_HT100to200", "QCD?") is True
        assert check_selector("QCD_HT100to200", "TTbar?") is False

    def test_list_selector(self):
        """Test matching against list of selectors."""
        from HH4b.utils import check_selector

        selectors = ["QCD?", "TTbar", "WJets?"]

        assert check_selector("QCD_HT100to200", selectors) is True
        assert check_selector("TTbar", selectors) is True
        assert check_selector("WJets_HT100to200", selectors) is True
        assert check_selector("ZJets", selectors) is False


class TestGetNevents:
    """Test get_nevents function."""

    def test_get_nevents_missing_directory(self):
        """Test handling of missing directory."""
        from HH4b.utils import get_nevents

        result = get_nevents("/nonexistent/path", "2022", "test_sample")
        assert result is None

    def test_get_nevents_valid_directory(self, mock_pickles_dir):
        """Test getting nevents from valid pickle directory."""
        from HH4b.utils import get_nevents

        nevents = get_nevents(str(mock_pickles_dir), "2022", "test_sample")

        # Should sum events from all pickle files: 100 + 200 + 300 = 600
        assert nevents == 600


class TestAddBoolArg:
    """Test add_bool_arg function."""

    def test_basic_bool_arg(self):
        """Test adding a basic boolean argument."""
        import argparse

        from HH4b.run_utils import add_bool_arg

        parser = argparse.ArgumentParser()
        add_bool_arg(parser, "test-flag", help="Test flag", default=False)

        # Test positive flag
        args = parser.parse_args(["--test-flag"])
        assert args.test_flag is True

        # Test negative flag
        args = parser.parse_args(["--no-test-flag"])
        assert args.test_flag is False

        # Test default
        args = parser.parse_args([])
        assert args.test_flag is False


class TestPathOperations:
    """Test path and file operations."""

    def test_create_temp_directory(self, temp_dir):
        """Test temporary directory creation."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Create a subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        assert subdir.exists()
        assert subdir.is_dir()

    def test_pickle_write_read(self, temp_dir):
        """Test writing and reading pickle files."""
        data = {"test": "value", "number": 42}

        pickle_path = temp_dir / "test.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(data, f)

        assert pickle_path.exists()

        with pickle_path.open("rb") as f:
            loaded_data = pickle.load(f)

        assert loaded_data == data


@pytest.mark.unit
class TestShapeVarBasic:
    """Basic tests for ShapeVar without histogram creation."""

    def test_shapevar_creation(self):
        """Test ShapeVar dataclass creation."""
        from HH4b.utils import ShapeVar

        shape_var = ShapeVar(
            var="pt", label="p_{T} [GeV]", bins=[10, 0, 100], reg=True
        )

        assert shape_var.var == "pt"
        assert shape_var.label == "p_{T} [GeV]"
        assert shape_var.reg is True

    def test_shapevar_blind_window(self):
        """Test ShapeVar with blind window."""
        from HH4b.utils import ShapeVar

        shape_var = ShapeVar(
            var="mjj",
            label="M_{jj} [GeV]",
            bins=[20, 0, 200],
            blind_window=[110, 140],
        )

        assert shape_var.blind_window == [110, 140]


@pytest.mark.unit
class TestSystBasic:
    """Basic tests for Syst dataclass."""

    def test_syst_creation(self):
        """Test Syst dataclass creation."""
        from HH4b.utils import Syst

        syst = Syst(samples=["ttbar"], label="ttbar_norm")

        assert syst.samples == ["ttbar"]
        assert syst.label == "ttbar_norm"
        assert len(syst.years) > 0  # Should have default years


@pytest.mark.integration
class TestPickleWorkflow:
    """Integration test for pickle workflow."""

    def test_create_and_read_pickles(self, temp_dir):
        """Test complete workflow of creating and reading pickle files."""
        # Create pickle files
        pickle_dir = temp_dir / "output"
        pickle_dir.mkdir()

        data = {
            "2022": {
                "QCD_HT100to200": {
                    "nevents": 1000,
                    "cutflow": {
                        "initial": 1000,
                        "trigger": 800,
                        "jet_selection": 600,
                        "final": 500,
                    },
                }
            }
        }

        pickle_path = pickle_dir / "test.pkl"
        with pickle_path.open("wb") as f:
            pickle.dump(data, f)

        # Read back
        with pickle_path.open("rb") as f:
            loaded_data = pickle.load(f)

        assert loaded_data == data
        assert loaded_data["2022"]["QCD_HT100to200"]["nevents"] == 1000

    def test_multiple_pickle_accumulation(self, temp_dir):
        """Test accumulating data from multiple pickle files."""
        from HH4b.utils import get_cutflow, get_nevents

        pickles_dir = temp_dir / "pickles"
        pickles_dir.mkdir()

        # Create multiple pickle files with consistent structure
        sample_name = "test_sample"
        year = "2022"

        for i in range(5):
            data = {
                year: {
                    sample_name: {
                        "nevents": 100 * (i + 1),
                        "cutflow": {
                            "initial": 100 * (i + 1),
                            "trigger": 80 * (i + 1),
                            "selection": 50 * (i + 1),
                        },
                    }
                }
            }
            pickle_path = pickles_dir / f"chunk_{i}.pkl"
            with pickle_path.open("wb") as f:
                pickle.dump(data, f)

        # Test nevents accumulation
        total_nevents = get_nevents(str(pickles_dir), year, sample_name)
        expected_nevents = sum(100 * (i + 1) for i in range(5))  # 100+200+300+400+500
        assert total_nevents == expected_nevents

        # Test cutflow accumulation
        cutflow = get_cutflow(str(pickles_dir), year, sample_name)
        assert cutflow["initial"] == expected_nevents
        assert cutflow["trigger"] == sum(80 * (i + 1) for i in range(5))
        assert cutflow["selection"] == sum(50 * (i + 1) for i in range(5))

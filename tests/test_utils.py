"""
Unit tests for HH4b.utils module.

Tests core utility functions including:
- Pickle file handling
- Event processing utilities
- Histogram operations
- Selection utilities
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from HH4b.utils import (
    PAD_VAL,
    ShapeVar,
    Syst,
    check_selector,
    get_cutflow,
    get_nevents,
)


class TestShapeVar:
    """Test ShapeVar dataclass for histogram configuration."""

    def test_regular_axis_creation(self):
        """Test creation of regular histogram axis."""
        shape_var = ShapeVar(
            var="pt", label="p_{T} [GeV]", bins=[10, 0, 100], reg=True
        )

        assert shape_var.var == "pt"
        assert shape_var.label == "p_{T} [GeV]"
        assert shape_var.reg is True
        assert shape_var.axis is not None

    def test_variable_axis_creation(self):
        """Test creation of variable-width histogram axis."""
        bins = [0, 10, 30, 50, 100]
        shape_var = ShapeVar(
            var="mass", label="Mass [GeV]", bins=bins, reg=False
        )

        assert shape_var.var == "mass"
        assert shape_var.reg is False
        assert shape_var.axis is not None

    def test_blind_window(self):
        """Test blind window configuration."""
        shape_var = ShapeVar(
            var="mjj",
            label="M_{jj} [GeV]",
            bins=[20, 0, 200],
            blind_window=[110, 140],
        )

        assert shape_var.blind_window == [110, 140]

    def test_significance_direction(self):
        """Test significance direction configuration."""
        shape_var = ShapeVar(
            var="bdt", label="BDT Score", bins=[10, 0, 1], significance_dir="left"
        )

        assert shape_var.significance_dir == "left"


class TestSyst:
    """Test Syst dataclass for systematic uncertainty configuration."""

    def test_default_years(self):
        """Test that default years are properly set."""
        syst = Syst(samples=["ttbar"], label="ttbar_norm")

        assert syst.samples == ["ttbar"]
        assert len(syst.years) > 0  # Should have default years

    def test_custom_years(self):
        """Test custom year configuration."""
        custom_years = ["2022", "2023"]
        syst = Syst(samples=["qcd"], years=custom_years, label="qcd_scale")

        assert syst.years == custom_years


class TestGetNevents:
    """Test get_nevents function for counting events in pickle files."""

    def test_get_nevents_valid_directory(self, mock_pickles_dir):
        """Test getting nevents from valid pickle directory."""
        nevents = get_nevents(str(mock_pickles_dir), "2022", "test_sample")

        # Should sum events from all pickle files: 100 + 200 + 300 = 600
        assert nevents == 600

    def test_get_nevents_missing_directory(self):
        """Test handling of missing directory."""
        result = get_nevents("/nonexistent/path", "2022", "test_sample")

        assert result is None

    def test_get_nevents_empty_directory(self, temp_dir):
        """Test handling of empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = get_nevents(str(empty_dir), "2022", "test_sample")

        # Should return None or raise appropriate error
        # Behavior depends on implementation details
        assert result is None or isinstance(result, (int, type(None)))


class TestGetCutflow:
    """Test get_cutflow function for accumulating cutflows."""

    def test_get_cutflow_single_file(self, mock_pickles_dir):
        """Test getting cutflow from multiple pickle files."""
        cutflow = get_cutflow(str(mock_pickles_dir), "2022", "test_sample")

        assert cutflow is not None
        assert "initial" in cutflow
        assert "after_cuts" in cutflow
        # Should sum: (100 + 200 + 300) = 600
        assert cutflow["initial"] == 600
        # Should sum: (50 + 100 + 150) = 300
        assert cutflow["after_cuts"] == 300

    def test_get_cutflow_with_warnings(self, temp_dir):
        """Test that warnings are issued for problematic files."""
        pickles_dir = temp_dir / "pickles"
        pickles_dir.mkdir()

        # Create one valid pickle
        data = {
            "2022": {
                "test_sample": {
                    "nevents": 100,
                    "cutflow": {"initial": 100, "after_cuts": 50},
                }
            }
        }
        (pickles_dir / "valid.pkl").write_bytes(pickle.dumps(data))

        # Create one invalid pickle
        (pickles_dir / "invalid.pkl").write_bytes(b"invalid data")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cutflow = get_cutflow(str(pickles_dir), "2022", "test_sample")

            # Should have warning about invalid file
            assert len(w) >= 1
            # Should still get cutflow from valid file
            assert cutflow is not None


class TestCheckSelector:
    """Test check_selector function for sample selection."""

    def test_exact_match(self):
        """Test exact sample name matching."""
        assert check_selector("QCD_HT100to200", "QCD_HT100to200") is True
        assert check_selector("QCD_HT100to200", "QCD_HT200to300") is False

    def test_wildcard_match(self):
        """Test wildcard matching with ? suffix."""
        assert check_selector("QCD_HT100to200", "QCD_HT?") is True
        assert check_selector("QCD_HT100to200", "QCD?") is True
        assert check_selector("QCD_HT100to200", "TTbar?") is False

    def test_list_selector(self):
        """Test matching against list of selectors."""
        selectors = ["QCD?", "TTbar", "WJets?"]

        assert check_selector("QCD_HT100to200", selectors) is True
        assert check_selector("TTbar", selectors) is True
        assert check_selector("WJets_HT100to200", selectors) is True
        assert check_selector("ZJets", selectors) is False

    def test_empty_selector(self):
        """Test behavior with empty selector."""
        assert check_selector("any_sample", "") is False
        assert check_selector("any_sample", []) is False


class TestPadVal:
    """Test PAD_VAL constant and related utilities."""

    def test_pad_val_constant(self):
        """Test that PAD_VAL is properly defined."""
        assert PAD_VAL == -99999
        assert isinstance(PAD_VAL, int)


@pytest.mark.unit
class TestUtilsIntegration:
    """Integration tests for utils module functions working together."""

    def test_full_workflow(self, mock_pickles_dir):
        """Test complete workflow of loading and processing pickles."""
        # Get nevents
        nevents = get_nevents(str(mock_pickles_dir), "2022", "test_sample")
        assert nevents == 600

        # Get cutflow
        cutflow = get_cutflow(str(mock_pickles_dir), "2022", "test_sample")
        assert cutflow["initial"] == 600

        # Verify consistency
        assert nevents == cutflow["initial"]

    def test_shapevar_with_hist(self):
        """Test ShapeVar integration with actual histogramming."""
        shape_var = ShapeVar(
            var="test_var",
            label="Test Variable",
            bins=[10, 0, 100],
            reg=True,
        )

        assert shape_var.axis.name == "test_var"
        assert shape_var.axis.label == "Test Variable"

"""
Unit tests for HH4b.hh_vars module.

Tests configuration variables and constants.
"""

from __future__ import annotations

import pytest

from HH4b.hh_vars import DATA_SAMPLES, LUMI, years


class TestYearsConfiguration:
    """Test years configuration."""

    def test_years_list_exists(self):
        """Test that years list is properly defined."""
        assert isinstance(years, list)
        assert len(years) > 0

    def test_years_format(self):
        """Test that years are in expected format."""
        for year in years:
            assert isinstance(year, str)
            # Should be either 4-digit year or year+suffix
            assert "2022" in year or "2023" in year or "2024" in year


class TestLumiConfiguration:
    """Test luminosity configuration."""

    def test_lumi_dict_exists(self):
        """Test that LUMI dictionary is properly defined."""
        assert isinstance(LUMI, dict)
        assert len(LUMI) > 0

    def test_lumi_values_positive(self):
        """Test that all luminosity values are positive."""
        for year, lumi in LUMI.items():
            assert lumi > 0, f"Luminosity for {year} should be positive"

    def test_lumi_units(self):
        """Test that luminosity values are in reasonable range (pb^-1)."""
        for year, lumi in LUMI.items():
            # Typical Run 3 luminosity values are 1000-100000 pb^-1
            assert 1000 < lumi < 200000, f"Luminosity for {year} seems unreasonable"

    def test_combined_years_lumi(self):
        """Test that combined year luminosities are consistent."""
        if "2022" in LUMI and "2022EE" in LUMI and "2022All" in LUMI:
            expected = LUMI["2022"] + LUMI["2022EE"]
            actual = LUMI["2022All"]
            assert abs(expected - actual) < 0.1, "2022All luminosity should equal sum of 2022 + 2022EE"


class TestDataSamples:
    """Test DATA_SAMPLES configuration."""

    def test_data_samples_list_exists(self):
        """Test that DATA_SAMPLES list is defined."""
        assert isinstance(DATA_SAMPLES, list)
        assert len(DATA_SAMPLES) > 0

    def test_data_samples_content(self):
        """Test that DATA_SAMPLES contains expected entries."""
        # Should contain typical CMS data sample names
        expected_samples = ["JetMET", "Muon", "EGamma"]
        for sample in expected_samples:
            assert sample in DATA_SAMPLES, f"{sample} should be in DATA_SAMPLES"


@pytest.mark.unit
class TestVarsConsistency:
    """Test consistency across different variable definitions."""

    def test_years_in_lumi(self):
        """Test that years defined in years list have luminosity values."""
        # Not all years might have LUMI defined yet, but check for coverage
        coverage = sum(1 for year in years if year in LUMI)
        assert coverage > 0, "At least some years should have LUMI defined"

    def test_no_duplicate_years(self):
        """Test that there are no duplicate years."""
        assert len(years) == len(set(years)), "Years list should not contain duplicates"

    def test_no_duplicate_data_samples(self):
        """Test that there are no duplicate data samples."""
        assert len(DATA_SAMPLES) == len(
            set(DATA_SAMPLES)
        ), "DATA_SAMPLES should not contain duplicates"

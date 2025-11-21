"""
Integration tests for HH4b workflows.

Tests end-to-end functionality including:
- File processing workflows
- Pickle handling
- Data transformations
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
class TestPickleWorkflow:
    """Integration tests for pickle file processing workflow."""

    def test_create_read_pickle_workflow(self, temp_dir):
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


@pytest.mark.integration
class TestDataProcessing:
    """Integration tests for data processing pipelines."""

    def test_event_filtering_workflow(self, mock_events_data):
        """Test filtering events through multiple selection criteria."""
        # Simulate event selection workflow
        events = mock_events_data

        # Selection 1: pt > 150
        pt_mask = events["pt"] > 150
        assert np.sum(pt_mask) == 3  # 3 events pass

        # Selection 2: |eta| < 0.8
        eta_mask = np.abs(events["eta"]) < 0.8
        assert np.sum(eta_mask) == 3  # 3 events pass

        # Combined selection
        combined_mask = pt_mask & eta_mask
        assert np.sum(combined_mask) == 2  # 2 events pass both

    def test_weight_application_workflow(self, mock_dataframe):
        """Test applying weights to events."""
        df = mock_dataframe

        # Check initial state
        assert len(df) == 5
        assert "weight" in df.columns

        # Apply additional weight
        df["total_weight"] = df["weight"] * 1.5

        # Verify
        assert df["total_weight"].iloc[0] == pytest.approx(1.5)
        assert df["total_weight"].iloc[1] == pytest.approx(1.65)


@pytest.mark.integration
class TestSampleSelection:
    """Integration tests for sample selection logic."""

    def test_sample_pattern_matching_workflow(self):
        """Test complete sample selection workflow."""
        from HH4b.utils import check_selector

        samples = [
            "QCD_HT100to200",
            "QCD_HT200to300",
            "TTto4Q",
            "TTto2L2Nu",
            "GluGlutoHHto4B_kl-1p00",
        ]

        # Select QCD samples
        qcd_samples = [s for s in samples if check_selector(s, "QCD?")]
        assert len(qcd_samples) == 2
        assert all("QCD" in s for s in qcd_samples)

        # Select TT samples
        tt_samples = [s for s in samples if check_selector(s, "TT?")]
        assert len(tt_samples) == 2
        assert all("TT" in s for s in tt_samples)

        # Select signal samples
        sig_samples = [s for s in samples if check_selector(s, "GluGlutoHHto4B?")]
        assert len(sig_samples) == 1

        # Select multiple patterns
        patterns = ["QCD?", "TT?"]
        selected = [s for s in samples if any(check_selector(s, p) for p in patterns)]
        assert len(selected) == 4


@pytest.mark.integration
@pytest.mark.slow
class TestFileOperations:
    """Integration tests for file operations."""

    def test_directory_creation_workflow(self, temp_dir):
        """Test directory creation and cleanup workflow."""
        # Create nested directories
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True, exist_ok=True)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

        # Create a file in nested directory
        test_file = nested_dir / "test.txt"
        test_file.write_text("test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_pickle_error_handling_workflow(self, temp_dir):
        """Test error handling in pickle operations."""
        from HH4b.utils import get_nevents

        # Test with nonexistent directory
        result = get_nevents("/nonexistent/path", "2022", "sample")
        assert result is None

        # Test with empty directory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        result = get_nevents(str(empty_dir), "2022", "sample")
        # Should handle gracefully
        assert result is None or isinstance(result, int)


@pytest.mark.integration
class TestConfigurationWorkflow:
    """Integration tests for configuration handling."""

    def test_year_lumi_configuration(self, sample_year_configs):
        """Test year and luminosity configuration workflow."""
        config = sample_year_configs

        # Verify years
        assert "2022" in config["years"]
        assert "2023" in config["years"]

        # Verify lumi values
        assert config["lumi"]["2022"] > 0
        assert config["lumi"]["2023"] > 0

        # Test total lumi calculation
        total_lumi = sum(config["lumi"].values())
        assert total_lumi > 0

    def test_cross_section_workflow(self, mock_xsecs):
        """Test cross-section handling workflow."""
        xsecs = mock_xsecs

        # Calculate effective cross-section
        for sample, info in xsecs.items():
            eff_xsec = info["xsec"] * info["kfactor"] * info["br"]
            assert eff_xsec > 0

            # Store for later use
            info["eff_xsec"] = eff_xsec

        # Signal should have smaller xsec than backgrounds
        assert xsecs["GluGlutoHHto4B"]["eff_xsec"] < xsecs["QCD_HT100to200"]["eff_xsec"]


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests."""

    def test_complete_analysis_workflow(self, temp_dir, mock_xsecs):
        """Test complete analysis workflow from data to results."""
        # Step 1: Create input data
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        sample_data = {
            "2022": {
                "QCD_HT100to200": {
                    "nevents": 1000000,
                    "cutflow": {
                        "initial": 1000000,
                        "trigger": 500000,
                        "selection": 100000,
                    },
                },
                "GluGlutoHHto4B": {
                    "nevents": 10000,
                    "cutflow": {
                        "initial": 10000,
                        "trigger": 8000,
                        "selection": 5000,
                    },
                },
            }
        }

        for sample in ["QCD_HT100to200", "GluGlutoHHto4B"]:
            sample_dir = input_dir / sample / "pickles"
            sample_dir.mkdir(parents=True)

            data = {
                "2022": {
                    sample: sample_data["2022"][sample]
                }
            }

            pickle_path = sample_dir / "data.pkl"
            with pickle_path.open("wb") as f:
                pickle.dump(data, f)

        # Step 2: Process samples
        from HH4b.utils import get_nevents

        results = {}
        for sample in ["QCD_HT100to200", "GluGlutoHHto4B"]:
            sample_dir = input_dir / sample / "pickles"
            nevents = get_nevents(str(sample_dir), "2022", sample)
            results[sample] = nevents

        # Step 3: Verify results
        assert results["QCD_HT100to200"] == 1000000
        assert results["GluGlutoHHto4B"] == 10000

        # Step 4: Calculate efficiencies
        for sample in ["QCD_HT100to200", "GluGlutoHHto4B"]:
            initial = sample_data["2022"][sample]["cutflow"]["initial"]
            final = sample_data["2022"][sample]["cutflow"]["selection"]
            efficiency = final / initial if initial > 0 else 0

            assert 0 <= efficiency <= 1

        # Signal should have higher efficiency
        qcd_eff = sample_data["2022"]["QCD_HT100to200"]["cutflow"]["selection"] / \
                  sample_data["2022"]["QCD_HT100to200"]["cutflow"]["initial"]
        sig_eff = sample_data["2022"]["GluGlutoHHto4B"]["cutflow"]["selection"] / \
                  sample_data["2022"]["GluGlutoHHto4B"]["cutflow"]["initial"]

        assert sig_eff > qcd_eff  # Signal should be more efficient

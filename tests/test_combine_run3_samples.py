"""Tests for the per-process lumi scaling in ``combine_run3_samples``.

When a process has dedicated MC for *every* year being combined, the
extrapolation must be a no-op (``lumi_scale == 1``) so that the per-year
templates sliced out of ``events_combined`` are not spuriously scaled up
to the full luminosity. Only a process genuinely missing MC for some era
should be extrapolated. This guards against the "MC-available years" list
going stale (e.g. omitting 2024/2025 after their MC was added), which
otherwise over-normalizes those eras' per-year templates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from HH4b.hh_vars import LUMI
from HH4b.postprocessing.postprocessing import combine_run3_samples


def _events(weight: float, n: int = 5, year: str = "2022") -> pd.DataFrame:
    return pd.DataFrame({"weight": np.full(n, float(weight)), "year": [year] * n})


def test_no_scale_when_all_years_available():
    """Process present in every combined year -> lumi_scale == 1, weights unchanged."""
    years = ["2022", "2022EE"]
    events = {y: {"ttbar": _events(1.0, year=y)} for y in years}

    combined, scaled_by = combine_run3_samples(
        events, ["ttbar"], scale_processes={"ttbar": years}, years_run3=years
    )

    assert np.isclose(scaled_by["ttbar"], 1.0)
    assert np.isclose(combined["ttbar"]["weight"].sum(), 10.0)  # 5 + 5, unscaled


def test_extrapolates_when_year_missing():
    """Process with MC only in 2022 while combining 2022+2022EE -> scaled to full lumi."""
    years = ["2022", "2022EE"]
    events = {"2022": {"ttbar": _events(1.0, year="2022")}, "2022EE": {}}

    combined, scaled_by = combine_run3_samples(
        events, ["ttbar"], scale_processes={"ttbar": ["2022"]}, years_run3=years
    )

    expected = (LUMI["2022"] + LUMI["2022EE"]) / LUMI["2022"]
    assert np.isclose(scaled_by["ttbar"], expected)
    assert np.isclose(combined["ttbar"]["weight"].sum(), 5.0 * expected)


def test_unlisted_process_concatenated_unscaled():
    """A process not in scale_processes is concatenated across years, never scaled."""
    years = ["2022", "2022EE"]
    events = {y: {"qcd": _events(2.0, year=y)} for y in years}

    combined, scaled_by = combine_run3_samples(
        events, ["qcd"], scale_processes={}, years_run3=years
    )

    assert "qcd" not in scaled_by
    assert np.isclose(combined["qcd"]["weight"].sum(), 20.0)  # 5*2 + 5*2

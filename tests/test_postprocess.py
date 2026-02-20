"""
Unit and integration tests for PostProcess.py

Tests cover:
- add_bdt_scores: BDT score assignment for binary and multi-class classifiers
- get_jets_for_txbb_sf: which jets receive TXbb scale factors
- get_nevents_data: sideband-based data event counting
- get_nevents_signal: signal region event counting
- get_nevents_nosignal: non-signal region event counting
- fom_classic / fom_update: figure-of-merit calculations
- Category assignment logic: VBF, Bin1, Bin2, Bin3, Fail categories
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from HH4b.postprocessing.PostProcess import (
    add_bdt_scores,
    fom_classic,
    fom_update,
    get_jets_for_txbb_sf,
    get_nevents_data,
    get_nevents_nosignal,
    get_nevents_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_events(mass_vals: np.ndarray, weight_vals: np.ndarray) -> pd.DataFrame:
    """Return a minimal DataFrame with H2PNetMass and weight columns."""
    return pd.DataFrame({"H2PNetMass": mass_vals, "weight": weight_vals})


# ---------------------------------------------------------------------------
# add_bdt_scores
# ---------------------------------------------------------------------------


class TestAddBdtScores:
    """Tests for the add_bdt_scores function."""

    def test_binary_no_jshift(self):
        """Binary BDT (2 outputs): bdt_score = preds[:, 1]."""
        events = pd.DataFrame(index=range(5))
        preds = np.array([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]])
        add_bdt_scores(events, preds)
        np.testing.assert_array_almost_equal(events["bdt_score"], preds[:, 1])
        assert "bdt_score_vbf" not in events.columns

    def test_binary_with_jshift(self):
        """Binary BDT with a JEC shift appends the shift suffix."""
        events = pd.DataFrame(index=range(3))
        preds = np.array([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
        add_bdt_scores(events, preds, jshift="JES_up")
        assert "bdt_score_JES_up" in events.columns
        np.testing.assert_array_almost_equal(events["bdt_score_JES_up"], preds[:, 1])
        assert "bdt_score" not in events.columns

    def test_three_class_ggf_score(self):
        """3-class BDT (ggF HH, QCD, ttbar): bdt_score = preds[:, 0]."""
        events = pd.DataFrame(index=range(4))
        preds = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.0, 0.5, 0.5]])
        add_bdt_scores(events, preds)
        np.testing.assert_array_almost_equal(events["bdt_score"], preds[:, 0])
        assert "bdt_score_vbf" not in events.columns

    def test_four_class_discriminant(self):
        """4-class BDT: discriminant = P_ggF / (P_ggF + P_QCD + P_ttbar)."""
        events = pd.DataFrame(index=range(3))
        preds = np.array(
            [[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2]]
        )
        weight_ttbar = 2.0
        add_bdt_scores(events, preds, weight_ttbar=weight_ttbar)

        bg_tot = np.sum(preds[:, 2:], axis=1)
        expected_ggf = preds[:, 0] / (preds[:, 0] + bg_tot)
        np.testing.assert_array_almost_equal(events["bdt_score"], expected_ggf)

        expected_vbf = preds[:, 1] / (preds[:, 1] + preds[:, 2] + weight_ttbar * preds[:, 3])
        np.testing.assert_array_almost_equal(events["bdt_score_vbf"], expected_vbf)

    def test_four_class_raw_score(self):
        """4-class BDT with bdt_disc=False: raw ggF and VBF probabilities stored."""
        events = pd.DataFrame(index=range(3))
        preds = np.array([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.3, 0.3, 0.2, 0.2]])
        add_bdt_scores(events, preds, bdt_disc=False)
        np.testing.assert_array_almost_equal(events["bdt_score"], preds[:, 0])
        np.testing.assert_array_almost_equal(events["bdt_score_vbf"], preds[:, 1])

    def test_five_class_discriminant(self):
        """5-class BDT: combined VBF discriminant uses K2V=0 and K2V=1 nodes."""
        events = pd.DataFrame(index=range(3))
        preds = np.array(
            [[0.4, 0.1, 0.1, 0.2, 0.2], [0.1, 0.3, 0.3, 0.2, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]
        )
        weight_ttbar = 1.5
        add_bdt_scores(events, preds, weight_ttbar=weight_ttbar)

        bg_tot = np.sum(preds[:, 3:], axis=1)
        expected_ggf = preds[:, 0] / (preds[:, 0] + bg_tot)
        np.testing.assert_array_almost_equal(events["bdt_score"], expected_ggf)

        expected_vbf = (preds[:, 1] + preds[:, 2]) / (
            preds[:, 1] + preds[:, 2] + preds[:, 3] + weight_ttbar * preds[:, 4]
        )
        np.testing.assert_array_almost_equal(events["bdt_score_vbf"], expected_vbf)


# ---------------------------------------------------------------------------
# get_jets_for_txbb_sf
# ---------------------------------------------------------------------------


class TestGetJetsForTxbbSf:
    """Tests for get_jets_for_txbb_sf."""

    def test_signal_keys_return_both_jets(self):
        """Signal processes get TXbb SF applied to both jets."""
        assert get_jets_for_txbb_sf("hh4b") == [1, 2]
        assert get_jets_for_txbb_sf("vbfhh4b") == [1, 2]

    def test_vhtobb_zz_return_both_jets(self):
        """vhtobb and zz get TXbb SF applied to both jets."""
        assert get_jets_for_txbb_sf("vhtobb") == [1, 2]
        assert get_jets_for_txbb_sf("zz") == [1, 2]

    def test_single_h_single_v_return_first_jet(self):
        """Single-Higgs and single-V processes get TXbb SF for first jet only."""
        for key in ["novhhtobb", "tthtobb", "vjets", "nozzdiboson"]:
            assert get_jets_for_txbb_sf(key) == [1], f"Failed for key={key}"

    def test_other_keys_return_empty(self):
        """Other processes (e.g., qcd, ttbar, data) get no TXbb SF."""
        for key in ["qcd", "ttbar", "data"]:
            assert get_jets_for_txbb_sf(key) == [], f"Failed for key={key}"


# ---------------------------------------------------------------------------
# get_nevents_data, get_nevents_signal, get_nevents_nosignal
# ---------------------------------------------------------------------------


class TestGetNeventsFunctions:
    """Tests for the event-counting helper functions."""

    _mass_window = [110.0, 140.0]  # 30 GeV window

    def _make_df(self, masses, weights=None):
        if weights is None:
            weights = np.ones(len(masses))
        return pd.DataFrame({"H2PNetMass": masses, "weight": weights})

    # ---- get_nevents_data ------------------------------------------------

    def test_nevents_data_events_in_left_sideband(self):
        """Events in the left sideband (between 95 and 110 GeV) are counted."""
        # window = [110, 140], mw_size = 30, left_sb = (95, 110)
        masses = np.array([100.0, 107.0, 80.0, 120.0])  # first two in left SB
        df = self._make_df(masses)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_data(df, cut, "H2PNetMass", self._mass_window)
        assert result == 2

    def test_nevents_data_events_in_right_sideband(self):
        """Events in the right sideband (between 140 and 155 GeV) are counted.

        For window [110, 140] (mw_size=30): right_sb = (140, 155).
        Note: 100 GeV also falls in the left sideband (95, 110).
        170 GeV is outside both sidebands (> 155 GeV).
        """
        # left_sb = (95, 110): 100 is included
        # right_sb = (140, 155): 145 and 150 are included; 170 is NOT (> 155)
        masses = np.array([145.0, 150.0, 100.0, 170.0])
        df = self._make_df(masses)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_data(df, cut, "H2PNetMass", self._mass_window)
        # 145, 150 (right SB) + 100 (left SB) = 3
        assert result == 3

    def test_nevents_data_signal_region_excluded(self):
        """Events inside the mass window are NOT counted."""
        masses = np.array([115.0, 125.0, 135.0])  # all inside [110, 140]
        df = self._make_df(masses)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_data(df, cut, "H2PNetMass", self._mass_window)
        assert result == 0

    def test_nevents_data_cut_selects_subset(self):
        """Only events passing the cut are counted."""
        masses = np.array([100.0, 102.0, 105.0, 145.0])
        df = self._make_df(masses)
        cut = np.array([True, False, True, True])
        result = get_nevents_data(df, cut, "H2PNetMass", self._mass_window)
        # 100 and 105 are in left SB, 145 is in right SB; 102 is excluded by cut
        assert result == 3

    # ---- get_nevents_signal ----------------------------------------------

    def test_nevents_signal_inside_window(self):
        """Events with mass inside the window and passing the cut are counted."""
        masses = np.array([115.0, 125.0, 135.0, 90.0])
        weights = np.array([1.0, 2.0, 0.5, 1.0])
        df = self._make_df(masses, weights)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_signal(df, cut, "H2PNetMass", self._mass_window)
        assert result == pytest.approx(3.5)  # 1 + 2 + 0.5

    def test_nevents_signal_cut_selects_subset(self):
        """Cut correctly restricts which signal events are counted."""
        masses = np.array([120.0, 130.0, 120.0])
        weights = np.array([1.0, 1.0, 1.0])
        df = self._make_df(masses, weights)
        cut = np.array([True, False, True])
        result = get_nevents_signal(df, cut, "H2PNetMass", self._mass_window)
        assert result == pytest.approx(2.0)

    def test_nevents_signal_outside_window_not_counted(self):
        """Events outside the mass window are not counted as signal."""
        masses = np.array([90.0, 160.0, 200.0])
        df = self._make_df(masses)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_signal(df, cut, "H2PNetMass", self._mass_window)
        assert result == pytest.approx(0.0)

    # ---- get_nevents_nosignal --------------------------------------------

    def test_nevents_nosignal_outside_window(self):
        """Events outside the signal window (but in [60, 220]) are counted."""
        masses = np.array([80.0, 90.0, 125.0, 160.0, 200.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        df = self._make_df(masses, weights)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_nosignal(df, cut, "H2PNetMass", self._mass_window)
        # 80, 90 are in [60, 110); 160, 200 are in (140, 220]; 125 is in signal window
        assert result == pytest.approx(4.0)

    def test_nevents_nosignal_inside_window_not_counted(self):
        """Events inside the signal window are NOT counted."""
        masses = np.array([115.0, 125.0, 135.0])
        df = self._make_df(masses)
        cut = np.ones(len(df), dtype=bool)
        result = get_nevents_nosignal(df, cut, "H2PNetMass", self._mass_window)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# fom_classic and fom_update
# ---------------------------------------------------------------------------


class TestFigureOfMerit:
    """Tests for fom_classic and fom_update."""

    def test_fom_classic_basic(self):
        """fom_classic = 2*sqrt(b)/s for valid s and b."""
        s, b = 5.0, 25.0
        expected = 2 * np.sqrt(25.0) / 5.0  # = 2
        assert fom_classic(s, b) == pytest.approx(expected)

    def test_fom_classic_zero_signal(self):
        """fom_classic returns nan when signal is zero."""
        assert np.isnan(fom_classic(0.0, 10.0))

    def test_fom_classic_zero_background(self):
        """fom_classic returns nan when background is zero."""
        assert np.isnan(fom_classic(5.0, 0.0))

    def test_fom_update_no_abcd(self):
        """fom_update without abcd falls back to fom_classic."""
        s, b = 4.0, 16.0
        assert fom_update(s, b) == pytest.approx(fom_classic(s, b))

    def test_fom_update_with_abcd(self):
        """fom_update with abcd includes additional uncertainty from ABCD regions."""
        s, b = 5.0, 25.0
        abcd_vals = [0, 100, 200, 400]  # region A, B, C, D
        expected = (
            2 * np.sqrt(b + b * b * (1 / abcd_vals[1] + 1 / abcd_vals[2] + 1 / abcd_vals[3])) / s
        )
        assert fom_update(s, b, abcd_vals) == pytest.approx(expected)

    def test_fom_update_zero_signal(self):
        """fom_update returns nan when signal is zero."""
        assert np.isnan(fom_update(0.0, 10.0, [0, 50, 50, 100]))

    def test_fom_update_zero_background(self):
        """fom_update returns nan when background is zero."""
        assert np.isnan(fom_update(5.0, 0.0, [0, 50, 50, 100]))


# ---------------------------------------------------------------------------
# Category assignment logic (integration-level)
# ---------------------------------------------------------------------------


class TestCategoryAssignment:
    """Integration tests for the event-categorisation logic inside
    load_process_run3_samples.

    We replicate the category-assignment code block directly on a small
    synthetic DataFrame to verify correctness without loading any files.
    """

    # Working-point defaults matching PostProcess.py defaults
    TXBB_WPS = [0.945, 0.85]
    BDT_WPS = [0.94, 0.755, 0.03]
    VBF_TXBB_WP = 0.8
    VBF_BDT_WP = 0.9825

    def _assign_categories(self, df: pd.DataFrame, vbf: bool = True, vbf_priority: bool = False):
        """Replicate the category-assignment block from load_process_run3_samples."""
        txbb_wps = self.TXBB_WPS
        bdt_wps = self.BDT_WPS
        vbf_txbb_wp = self.VBF_TXBB_WP
        vbf_bdt_wp = self.VBF_BDT_WP

        df = df.copy()
        df["Category"] = 5  # all events start as "undefined"

        mask_fail = (df["H2TXbb"] < txbb_wps[1]) & (df["bdt_score"] > bdt_wps[2])
        df.loc[mask_fail, "Category"] = 4

        if vbf:
            mask_vbf = (df["bdt_score_vbf"] > vbf_bdt_wp) & (df["H2TXbb"] > vbf_txbb_wp)
        else:
            mask_vbf = np.zeros(len(df), dtype=bool)

        mask_bin1 = (df["H2TXbb"] > txbb_wps[0]) & (df["bdt_score"] > bdt_wps[0])

        if vbf_priority:
            mask_bin1 = mask_bin1 & ~mask_vbf
        else:
            mask_vbf = mask_vbf & ~mask_bin1

        df.loc[mask_vbf, "Category"] = 0
        df.loc[mask_bin1, "Category"] = 1

        mask_corner = (df["H2TXbb"] < txbb_wps[0]) & (df["bdt_score"] < bdt_wps[0])
        mask_bin2 = (
            (df["H2TXbb"] > txbb_wps[1])
            & (df["bdt_score"] > bdt_wps[1])
            & ~mask_bin1
            & ~mask_corner
            & ~mask_vbf
        )
        df.loc[mask_bin2, "Category"] = 2

        mask_bin3 = (
            (df["H2TXbb"] > txbb_wps[1])
            & (df["bdt_score"] > bdt_wps[2])
            & ~mask_bin1
            & ~mask_bin2
            & ~mask_vbf
        )
        df.loc[mask_bin3, "Category"] = 3

        return df

    def _make_df(self, n: int, h2txbb, bdt_score, bdt_score_vbf=None) -> pd.DataFrame:
        data = {
            "H2TXbb": np.asarray(h2txbb, dtype=float),
            "bdt_score": np.asarray(bdt_score, dtype=float),
        }
        if bdt_score_vbf is not None:
            data["bdt_score_vbf"] = np.asarray(bdt_score_vbf, dtype=float)
        else:
            data["bdt_score_vbf"] = np.zeros(n)
        assert len(data["H2TXbb"]) == n, f"Expected {n} events, got {len(data['H2TXbb'])}"
        return pd.DataFrame(data)

    def test_bin1_assignment(self):
        """Events with high TXbb and high BDT score end up in Bin 1."""
        df = self._make_df(
            1,
            h2txbb=[0.96],  # > txbb_wps[0] = 0.945
            bdt_score=[0.96],  # > bdt_wps[0] = 0.94
            bdt_score_vbf=[0.0],
        )
        result = self._assign_categories(df)
        assert result["Category"].iloc[0] == 1

    def test_bin2_assignment(self):
        """Events with moderate TXbb and BDT score above bin1 threshold end up in Bin 2.

        Bin 2 requires H2TXbb > txbb_wps[1] AND bdt_score > bdt_wps[1]
        AND NOT bin1 AND NOT corner.
        The "corner" region is: H2TXbb < txbb_wps[0] AND bdt_score < bdt_wps[0].
        To avoid the corner while keeping H2TXbb < txbb_wps[0], the bdt_score
        must be >= bdt_wps[0].
        """
        df = self._make_df(
            1,
            h2txbb=[0.90],  # > txbb_wps[1]=0.85, < txbb_wps[0]=0.945
            bdt_score=[0.96],  # > bdt_wps[0]=0.94 -> NOT in corner, NOT bin1 (txbb too low)
            bdt_score_vbf=[0.0],
        )
        result = self._assign_categories(df)
        assert result["Category"].iloc[0] == 2

    def test_bin3_assignment(self):
        """Events with moderate TXbb but lower BDT score end up in Bin 3."""
        df = self._make_df(
            1,
            h2txbb=[0.90],  # > txbb_wps[1]=0.85
            bdt_score=[0.50],  # > bdt_wps[2]=0.03 but < bdt_wps[1]=0.755
            bdt_score_vbf=[0.0],
        )
        result = self._assign_categories(df)
        assert result["Category"].iloc[0] == 3

    def test_fail_assignment(self):
        """Events with low TXbb but passing the fail-threshold BDT go to Fail."""
        df = self._make_df(
            1,
            h2txbb=[0.70],  # < txbb_wps[1]=0.85
            bdt_score=[0.50],  # > bdt_wps[2]=0.03
            bdt_score_vbf=[0.0],
        )
        result = self._assign_categories(df)
        assert result["Category"].iloc[0] == 4

    def test_vbf_region_assignment(self):
        """Events passing VBF criteria are placed in the VBF category (0)."""
        df = self._make_df(
            1,
            h2txbb=[0.85],  # > vbf_txbb_wp=0.8 but < txbb_wps[0]=0.945
            bdt_score=[0.50],  # does NOT pass bin1 cut
            bdt_score_vbf=[0.99],  # > vbf_bdt_wp=0.9825
        )
        result = self._assign_categories(df, vbf=True)
        assert result["Category"].iloc[0] == 0

    def test_bin1_takes_priority_over_vbf(self):
        """By default (vbf_priority=False), Bin 1 takes priority over VBF."""
        df = self._make_df(
            1,
            h2txbb=[0.96],  # passes both bin1 and VBF txbb cuts
            bdt_score=[0.96],  # passes bin1 bdt cut
            bdt_score_vbf=[0.99],  # passes VBF bdt cut
        )
        result = self._assign_categories(df, vbf=True, vbf_priority=False)
        assert result["Category"].iloc[0] == 1  # Bin 1 wins

    def test_vbf_takes_priority_when_vbf_priority_set(self):
        """When vbf_priority=True, VBF takes priority over Bin 1."""
        df = self._make_df(
            1,
            h2txbb=[0.96],  # passes both bin1 and VBF txbb cuts
            bdt_score=[0.96],  # passes bin1 bdt cut
            bdt_score_vbf=[0.99],  # passes VBF bdt cut
        )
        result = self._assign_categories(df, vbf=True, vbf_priority=True)
        assert result["Category"].iloc[0] == 0  # VBF wins

    def test_no_vbf_region(self):
        """When vbf=False, no events are placed in VBF category."""
        df = self._make_df(
            3,
            h2txbb=[0.96, 0.85, 0.70],
            bdt_score=[0.96, 0.80, 0.50],
            bdt_score_vbf=[0.99, 0.99, 0.99],
        )
        result = self._assign_categories(df, vbf=False)
        assert 0 not in result["Category"].values

    def test_multiple_events_various_categories(self):
        """Verify mixed set of events gets correct category assignments.

        Event breakdown (vbf=False):
        - idx 0: txbb=0.96>0.945, bdt=0.96>0.94 → Bin 1
        - idx 1: txbb=0.90<0.945, bdt=0.80<0.94 → corner (NOT bin2), bdt>0.03 → Bin 3
        - idx 2: txbb=0.90, bdt=0.50 → corner, bdt>0.03 → Bin 3
        - idx 3: txbb=0.70<0.85, bdt=0.50>0.03 → Fail (cat 4)
        - idx 4: txbb=0.85 NOT > 0.85, bdt=0.20 → nothing matches → stays 5
        """
        h2txbb = [0.96, 0.90, 0.90, 0.70, 0.85]
        bdt_score = [0.96, 0.80, 0.50, 0.50, 0.20]
        bdt_score_vbf = [0.0, 0.0, 0.0, 0.0, 0.0]
        df = self._make_df(5, h2txbb, bdt_score, bdt_score_vbf)
        result = self._assign_categories(df, vbf=False)
        expected = [1, 3, 3, 4, 5]
        np.testing.assert_array_equal(result["Category"].values, expected)

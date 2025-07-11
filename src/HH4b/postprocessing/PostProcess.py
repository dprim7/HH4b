from __future__ import annotations

import argparse
import importlib
import logging
import logging.config
import pprint
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import awkward as ak
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b import hh_vars, plotting, postprocessing, run_utils
from HH4b.hh_vars import (
    bg_keys,
    mreg_strings,
    samples_run3,
    ttbarsfs_decorr_ggfbdt_bins,
    ttbarsfs_decorr_txbb_bins,
    ttbarsfs_decorr_vbfbdt_bins,
    txbb_strings,
    txbbsfs_decorr_pt_bins,
    txbbsfs_decorr_txbb_wps,
)
from HH4b.log_utils import log_config
from HH4b.postprocessing import (
    Region,
    combine_run3_samples,
    corrections,
    get_weight_shifts,
    load_run3_samples,
)
from HH4b.utils import (
    ShapeVar,
    check_get_jec_var,
    discretize_var,
    get_var_mapping,
    singleVarHist,
)

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)

# get top-level HH4b directory
HH4B_DIR = Path(__file__).resolve().parents[3]

plt.style.use(hep.style.CMS)
hep.style.use("CMS")

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))

mpl.rcParams["font.size"] = 30
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["grid.color"] = "#CCCCCC"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["figure.dpi"] = 400
mpl.rcParams["figure.edgecolor"] = "none"

# modify samples run3
for year in samples_run3:
    samples_run3[year]["qcd"] = [
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",
        # "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
    ]

selection_regions = {
    "pass_vbf": Region(
        cuts={
            "Category": [0, 1],
        },
        label="VBF",
    ),
    "pass_bin1": Region(
        cuts={
            "Category": [1, 2],
        },
        label="Bin1",
    ),
    "pass_bin2": Region(
        cuts={
            "Category": [2, 3],
        },
        label="Bin2",
    ),
    "pass_bin3": Region(
        cuts={
            "Category": [3, 4],
        },
        label="Bin3",
    ),
    "fail": Region(
        cuts={
            "Category": [4, 5],
        },
        label="Fail",
    ),
}


label_by_mass = {
    "H2Msd": r"$m^{2}_\mathrm{SD}$ (GeV)",
    "H2PNetMass": r"$m^{2}_\mathrm{reg}$ (GeV)",
}


def get_jets_for_txbb_sf(key: str):
    # TODO: correct application of bb-tagging SF based on gen-matching to H(bb) or Z(bb)
    # for now, assuming mostly V=Z(bb) passes selection
    # apply to both jets in HH, VH, VV processes
    # apply to only first jet in single-H or single-V processes
    if key in hh_vars.sig_keys or key in ["vhtobb", "zz"]:
        return [1, 2]
    elif key in ["novhhtobb", "tthtobb", "vjets", "nozzdiboson"]:
        return [1]
    else:
        return []


def get_bdt_training_keys(bdt_model: str):
    inferences_dir = Path(
        f"{HH4B_DIR}/src/HH4b/boosted/bdt_trainings_run3/{bdt_model}/inferences/2022EE"
    )

    training_keys = []
    for child in inferences_dir.iterdir():
        if child.suffix == ".npy":
            training_keys.append(child.stem.split("evt_")[-1])

    logger.info(f"Found BDT Training keys {training_keys}")
    return training_keys


def add_bdt_scores(
    events: pd.DataFrame,
    preds: np.ArrayLike,
    jshift: str = "",
    weight_ttbar: float = 1,
    bdt_disc: bool = True,
):
    jlabel = "" if jshift == "" else "_" + jshift

    if preds.shape[1] == 2:  # binary BDT only
        events[f"bdt_score{jlabel}"] = preds[:, 1]
    elif preds.shape[1] == 3:  # multi-class BDT with ggF HH, QCD, ttbar classes
        events[f"bdt_score{jlabel}"] = preds[:, 0]  # ggF HH
    elif preds.shape[1] == 4:  # multi-class BDT with ggF HH, VBF HH, QCD, ttbar classes
        bg_tot = np.sum(preds[:, 2:], axis=1)
        events[f"bdt_score{jlabel}"] = (
            preds[:, 0] / (preds[:, 0] + bg_tot) if bdt_disc else preds[:, 0]
        )
        events[f"bdt_score_vbf{jlabel}"] = (
            preds[:, 1] / (preds[:, 1] + preds[:, 2] + weight_ttbar * preds[:, 3])
            if bdt_disc
            else preds[:, 1]
        )
    elif (
        preds.shape[1] == 5
    ):  # multi-class BDT with ggF HH, VBF HH(K2V=0), VBF HH(K2V=1), QCD, ttbar classes
        bg_tot = np.sum(preds[:, 3:], axis=1)
        events[f"bdt_score{jlabel}"] = (
            preds[:, 0] / (preds[:, 0] + bg_tot) if bdt_disc else preds[:, 0]
        )
        events[f"bdt_score_vbf{jlabel}"] = (
            (preds[:, 1] + preds[:, 2])
            / (preds[:, 1] + preds[:, 2] + preds[:, 3] + weight_ttbar * preds[:, 4])
            if bdt_disc
            else preds[:, 1] + preds[:, 2]
        )


def bdt_roc(events_combined: dict[str, pd.DataFrame], plot_dir: str, txbb_version: str, jshift=""):
    sig_keys = [
        "hh4b",
        "hh4b-kl0",
        "hh4b-kl2p45",
        "hh4b-kl5",
        "vbfhh4b",
        "vbfhh4b-k2v0",
    ]
    scores_keys = {
        "hh4b": "bdt_score",
        "hh4b-kl0": "bdt_score",
        "hh4b-kl2p45": "bdt_score",
        "hh4b-kl5": "bdt_score",
        "vbfhh4b": "bdt_score_vbf",
        "vbfhh4b-k2v0": "bdt_score_vbf",
    }
    bkg_keys = ["qcd", "ttbar"]

    def get_legtitle(txbb_str):
        title = r"FatJet p$_T^{(0,1)}$ > 250 GeV" + "\n"
        if "part" in txbb_str.lower():
            title += "$T_{Xbb}^{0}$ > 0.3"
        else:
            title += "$T_{Xbb}^{0}$ > 0.8"

        if "legacy" in txbb_str.lower():
            title += "\n" + "PNet Legacy"
        elif "part" in txbb_str.lower():
            title += "\n" + "GloParTv2"
        else:
            title += "\n" + "PNet 103X"

        title += "\n" + r"m$_{reg}$ > 50 GeV"
        title += "\n" + r"m$_{SD}^{0}$ > 40 GeV"

        return title

    legtitle = get_legtitle(txbb_version)

    if "bdt_score_vbf" not in events_combined["ttbar"]:
        sig_keys.remove("vbfhh4b-k2v0")

    for sig_key in sig_keys:
        rocs = postprocessing.make_rocs(
            events_combined,
            scores_keys[sig_key],
            "weight",
            sig_key,
            bkg_keys,
            jshift,
        )
        bkg_colors = {**plotting.color_by_sample, "merged": "orange"}
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        for bg_key in [*bkg_keys, "merged"]:
            ax.plot(
                rocs[bg_key]["tpr"],
                rocs[bg_key]["fpr"],
                linewidth=2,
                color=bkg_colors[bg_key],
                label=rocs[bg_key]["label"],
            )

        ax.set_xlim([0.0, 0.6])
        ax.set_ylim([1e-5, 1e-1])
        ax.set_yscale("log")

        ax.set_title(f"{plotting.label_by_sample[sig_key]} BDT ROC Curve")
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background efficiency")

        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")
        ax.legend(
            title=legtitle,
            bbox_to_anchor=(1.03, 1),
            loc="upper left",
        )
        fig.tight_layout()
        _jshift = f"_{jshift}" if jshift != "" else ""
        fig.savefig(plot_dir / f"{sig_key}_roc{_jshift}.png")
        fig.savefig(plot_dir / f"{sig_key}_roc{_jshift}.pdf", bbox_inches="tight")
        plt.close()

    bdt_axis = hist.axis.Regular(40, 0, 1, name="bdt", label=r"BDT")
    cat_axis = hist.axis.StrCategory([], name="cat", label="cat", growth=True)
    h_bdt = hist.Hist(bdt_axis, cat_axis)
    for sig_key in sig_keys:
        h_bdt.fill(
            events_combined[sig_key][scores_keys[sig_key]],
            sig_key,
            weight=events_combined[sig_key]["weight"],
        )

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    th_colours = ["#9381FF", "#1f78b4", "#a6cee3", "cyan", "blue"]

    for vbf_in_sig_key in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        # add lines at BDT cuts
        plot_thresholds = [0.88, 0.98] if vbf_in_sig_key else [0.98]

        isig = 0
        for sig_key in sig_keys:
            if ("vbf" in sig_key) == vbf_in_sig_key:
                continue
            rocs = postprocessing.make_rocs(
                events_combined, scores_keys[sig_key], "weight", sig_key, bkg_keys
            )
            pths = {th: [[], []] for th in plot_thresholds}
            for th in plot_thresholds:
                idx = find_nearest(rocs["merged"]["thresholds"], th)
                pths[th][0].append(rocs["merged"]["tpr"][idx])
                pths[th][1].append(rocs["merged"]["fpr"][idx])
            for k, th in enumerate(plot_thresholds):
                if isig == 0:
                    ax.scatter(
                        *pths[th],
                        marker="o",
                        s=40,
                        label=rf"BDT > {th}",
                        color=th_colours[k],
                        zorder=100,
                    )
                else:
                    ax.scatter(
                        *pths[th],
                        marker="o",
                        s=40,
                        color=th_colours[k],
                        zorder=100,
                    )

            ax.plot(
                rocs["merged"]["tpr"],
                rocs["merged"]["fpr"],
                linewidth=2,
                color=plotting.color_by_sample[sig_key],
                label=plotting.label_by_sample[sig_key],
            )
            isig = isig + 1
        ax.set_xlim([0.0, 0.6])
        ax.set_ylim([1e-5, 1e-1])
        ax.set_yscale("log")
        if vbf_in_sig_key:
            ax.set_title("ggF BDT ROC Curve")
        else:
            ax.set_title("VBF BDT ROC Curve")
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background efficiency")
        ax.xaxis.grid(True, which="major")
        ax.yaxis.grid(True, which="major")

        ax.legend(
            title=legtitle,
            bbox_to_anchor=(1.03, 1),
            loc="upper left",
        )
        fig.tight_layout()
        if vbf_in_sig_key:
            fig.savefig(plot_dir / f"GGF_hh4b_allroc{_jshift}.png", bbox_inches="tight")
            fig.savefig(plot_dir / f"GGF_hh4b_allroc{_jshift}.pdf", bbox_inches="tight")
        else:
            fig.savefig(plot_dir / f"VBF_hh4b_allroc{_jshift}.png", bbox_inches="tight")
            fig.savefig(plot_dir / f"VBF_hh4b_allroc{_jshift}.pdf", bbox_inches="tight")
        plt.close()

        # plot scores too
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        for sig_key in sig_keys:
            if ("vbf" in sig_key) == vbf_in_sig_key:
                continue
            hep.histplot(
                h_bdt[{"cat": sig_key}],
                ax=ax,
                label=plotting.label_by_sample[sig_key],
                color=plotting.color_by_sample[sig_key],
                histtype="step",
                linewidth=1.5,
                density=True,
                flow="none",
            )
        ax.legend()
        fig.tight_layout()
        if vbf_in_sig_key:
            fig.savefig(plot_dir / f"GGF_hh4b_allbdt{_jshift}.png", bbox_inches="tight")
            fig.savefig(plot_dir / f"GGF_hh4b_allbdt{_jshift}.pdf", bbox_inches="tight")
        else:
            fig.savefig(plot_dir / f"VBF_hh4b_allbdt{_jshift}.png", bbox_inches="tight")
            fig.savefig(plot_dir / f"VBF_hh4b_allbdt{_jshift}.pdf", bbox_inches="tight")
        plt.close()


def load_process_run3_samples(
    args, year, bdt_training_keys, control_plots, plot_dir, mass_window, rerun_inference=False
):
    plot_dir = Path(plot_dir)
    # define BDT model
    if rerun_inference:
        bdt_model = xgb.XGBClassifier()
        bdt_model.load_model(
            fname=f"{HH4B_DIR}/src/HH4b/boosted/bdt_trainings_run3/{args.bdt_model}/trained_bdt.model"
        )

    # load tt corrections
    tt_ptjj_sf = corrections._load_ttbar_sfs(year, "PTJJ", args.txbb)
    tt_xbb_sf = corrections._load_ttbar_sfs(year, "Xbb", args.txbb)
    tt_tau32_sf = corrections._load_ttbar_sfs(year, "Tau3OverTau2", args.txbb)
    tt_ggfbdtshape_sf = corrections._load_ttbar_bdtshape_sfs(
        "cat5",
        (
            args.bdt_model
            if args.bdt_model in ttbarsfs_decorr_ggfbdt_bins
            else "25Feb5_v13_glopartv2_rawmass"
        ),
        "bdt_score",
    )
    tt_vbfbdtshape_sf = corrections._load_ttbar_bdtshape_sfs(
        "cat5",
        (
            args.bdt_model
            if args.bdt_model in ttbarsfs_decorr_vbfbdt_bins
            else "25Feb5_v13_glopartv2_rawmass"
        ),
        "bdt_score_vbf",
    )

    # get dictionary bins from keys
    # add defaults so that these do not fail
    ttsf_xbb_bins = ttbarsfs_decorr_txbb_bins.get(
        args.txbb, ttbarsfs_decorr_txbb_bins["glopart-v2"]
    )
    ttsf_ggfbdtshape_bins = ttbarsfs_decorr_ggfbdt_bins.get(
        args.bdt_model, ttbarsfs_decorr_ggfbdt_bins["25Feb5_v13_glopartv2_rawmass"]
    )
    if args.correct_vbf_bdt_shape:
        ttsf_vbfbdtshape_bins = ttbarsfs_decorr_vbfbdt_bins.get(
            args.bdt_model, ttbarsfs_decorr_vbfbdt_bins["25Feb5_v13_glopartv2_rawmass"]
        )
    TXbb_pt_corr_bins = txbbsfs_decorr_pt_bins.get(args.txbb, txbbsfs_decorr_pt_bins["glopart-v2"])
    TXbb_wps = txbbsfs_decorr_txbb_wps.get(args.txbb, txbbsfs_decorr_txbb_wps["glopart-v2"])

    # load TXbb SFs
    if args.dummy_txbb_sfs or args.txbb not in ["pnet-legacy", "glopart-v2"]:
        txbb_sf = corrections._load_dummy_txbb_sfs(
            TXbb_wps,
            TXbb_pt_corr_bins,
            sf=1.0,
            sf_unc=0.15,
        )
    elif args.txbb == "pnet-legacy":
        txbb_sf = corrections._load_txbb_sfs(
            year,
            "sf_txbbv11_Jul3_freezeSFs_combinedWPs",
            TXbb_wps,
            TXbb_pt_corr_bins,
            args.txbb,
        )
    elif args.txbb == "glopart-v2":
        txbb_sf = corrections._load_txbb_sfs(
            year,
            "sf_glopart-v2_zbb",
            TXbb_wps,
            TXbb_pt_corr_bins,
            args.txbb,
        )

    # get function
    make_bdt_dataframe = importlib.import_module(
        f".{args.bdt_config}", package="HH4b.boosted.bdt_trainings_run3"
    )

    # define cutflows
    samples_year = list(samples_run3[year].keys())
    if not control_plots and not args.bdt_roc:
        samples_year.remove("qcd")
    cutflow = pd.DataFrame(index=samples_year)
    cutflow_dict = {}

    # region in which QCD trigger weights were extracted
    trigger_region = "QCD"

    events_dict_postprocess = {}
    columns_by_key = {}
    # if year == "2024":
    #    samples_year.remove("qcd")
    for key in samples_year:
        logger.info(f"Load samples {key}")

        samples_to_process = {year: {key: samples_run3[year][key]}}

        events_dict = load_run3_samples(
            f"{args.data_dir}/{args.tag}",
            year,
            samples_to_process,
            reorder_txbb=True,
            load_systematics=True,
            txbb_version=args.txbb,
            scale_and_smear=args.scale_smear,
            mass_str=mreg_strings[args.txbb],
            bdt_version=args.bdt_model,
        )[key]

        # inference and assign score
        jshifts = [""]
        if key in hh_vars.syst_keys:
            jshifts += hh_vars.jec_shifts
        if key in hh_vars.jmsr_keys:
            jshifts += hh_vars.jmsr_shifts
        logger.info(f"JEC shifts {jshifts}")

        logger.info("Add BDT scores")
        bdt_events = {}
        for jshift in jshifts:
            _jshift = f"_{jshift}" if jshift != "" else ""
            bdt_events[jshift] = make_bdt_dataframe.bdt_dataframe(
                events_dict, get_var_mapping(jshift)
            )
            if rerun_inference:
                print("Re-run inference")
                preds = bdt_model.predict_proba(bdt_events[jshift])
                add_bdt_scores(
                    bdt_events[jshift],
                    preds,
                    jshift,
                    weight_ttbar=args.weight_ttbar_bdt,
                    bdt_disc=args.bdt_disc,
                )
            else:
                # assert bdt_disc is true
                if not args.bdt_disc:
                    raise ValueError("only BDT discriminant available from skimmer")
                bdt_events[jshift][f"bdt_score{_jshift}"] = events_dict[f"bdt_score{_jshift}"]
                bdt_events[jshift][f"bdt_score_vbf{_jshift}"] = events_dict[
                    f"bdt_score_vbf{_jshift}"
                ]

            # redefine VBF variables
            key_map = get_var_mapping(jshift)
            vbf1_pt = events_dict[(key_map("VBFJetPt"), 0)]
            vbf2_pt = events_dict[(key_map("VBFJetPt"), 1)]
            mask_negative_vbf = (vbf1_pt < 0) | (vbf2_pt < 0)
            bdt_events[jshift].loc[mask_negative_vbf, key_map("VBFjjMass")] = -1
            bdt_events[jshift].loc[mask_negative_vbf, key_map("VBFjjDeltaEta")] = -1

            # redefine AK4Away variables
            ak4away1_pt = events_dict[(key_map("AK4JetAwayPt"), 0)]
            ak4away2_pt = events_dict[(key_map("AK4JetAwayPt"), 1)]
            mask_negative_ak4away1 = ak4away1_pt < 0
            mask_negative_ak4away2 = ak4away2_pt < 0
            bdt_events[jshift].loc[mask_negative_ak4away1, key_map("H1AK4JetAway1dR")] = -1
            bdt_events[jshift].loc[mask_negative_ak4away1, key_map("H1AK4JetAway1mass")] = -1
            bdt_events[jshift].loc[mask_negative_ak4away2, key_map("H2AK4JetAway2dR")] = -1
            bdt_events[jshift].loc[mask_negative_ak4away2, key_map("H2AK4JetAway2mass")] = -1

        bdt_events = pd.concat([bdt_events[jshift] for jshift in jshifts], axis=1)

        # remove duplicates
        bdt_events = bdt_events.loc[:, ~bdt_events.columns.duplicated()].copy()

        # add more variables for control plots
        bdt_events["H1Pt"] = events_dict["bbFatJetPt"][0]
        bdt_events["H2Pt"] = events_dict["bbFatJetPt"][1]
        bdt_events["H1Msd"] = events_dict["bbFatJetMsd"][0]
        bdt_events["H2Msd"] = events_dict["bbFatJetMsd"][1]
        bdt_events["H1TXbb"] = events_dict[txbb_strings[args.txbb]][0]
        bdt_events["H1DiscTXbb"] = discretize_var(
            events_dict[txbb_strings[args.txbb]][0], bins=[0, 0.8, 0.9, 0.94, 0.97, 0.99, 1]
        )
        bdt_events["H2TXbb"] = events_dict[txbb_strings[args.txbb]][1]
        bdt_events["H2DiscTXbb"] = discretize_var(
            events_dict[txbb_strings[args.txbb]][1], bins=[0, 0.8, 0.9, 0.94, 0.97, 0.99, 1]
        )
        bdt_events["H1PNetMass"] = events_dict[mreg_strings[args.txbb]][0]
        bdt_events["H2PNetMass"] = events_dict[mreg_strings[args.txbb]][1]
        if key in hh_vars.jmsr_keys:
            for jshift in hh_vars.jmsr_shifts:
                bdt_events[f"H1PNetMass_{jshift}"] = events_dict[
                    f"{mreg_strings[args.txbb]}_{jshift}"
                ][0]
                bdt_events[f"H2PNetMass_{jshift}"] = events_dict[
                    f"{mreg_strings[args.txbb]}_{jshift}"
                ][1]

        # add HLTs
        bdt_events["hlt"] = np.any(
            np.array(
                [
                    events_dict[trigger].to_numpy()[:, 0]
                    for trigger in postprocessing.HLTs[year]
                    if trigger in events_dict
                ]
            ),
            axis=0,
        )

        # finalWeight: includes genWeight, puWeight
        bdt_events["weight"] = events_dict["finalWeight"].to_numpy()
        # scale, pdf weights
        if key in hh_vars.sig_keys + ["ttbar"]:
            for i in range(6):
                bdt_events[f"scale_weights_{i}"] = events_dict["scale_weights"][i].to_numpy()
        n_pdf_weights = 0
        if key in hh_vars.sig_keys:
            n_pdf_weights = events_dict["pdf_weights"].shape[1]
            for i in range(n_pdf_weights):
                bdt_events[f"pdf_weights_{i}"] = events_dict["pdf_weights"][i].to_numpy()
        pileup_ps_weights = [
            "weight_pileupUp",
            "weight_pileupDown",
            "weight_ISRPartonShowerUp",
            "weight_ISRPartonShowerDown",
            "weight_FSRPartonShowerUp",
            "weight_FSRPartonShowerDown",
        ]
        if key != "data":
            for w in pileup_ps_weights:
                if w in events_dict:
                    bdt_events[w] = events_dict[w].to_numpy()

        # add event, run, lumi
        bdt_events["run"] = events_dict["run"].to_numpy()
        bdt_events["event"] = events_dict["event"].to_numpy()
        bdt_events["luminosityBlock"] = events_dict["luminosityBlock"].to_numpy()

        # triggerWeight
        nevents = len(bdt_events["H1Pt"])
        trigger_weight = np.ones(nevents)
        trigger_weight_up = np.ones(nevents)
        trigger_weight_dn = np.ones(nevents)
        if key != "data":
            trigger_weight, _, total, total_err = corrections.trigger_SF(
                year, events_dict, txbb_strings[args.txbb], trigger_region
            )
            trigger_weight_up = trigger_weight * (1 + total_err / total)
            trigger_weight_dn = trigger_weight * (1 - total_err / total)

        # TXbbWeight
        txbb_sf_weight = np.ones(nevents)
        all_txbb_bins = ak.Array([TXbb_wps[wp] for wp in TXbb_wps])
        all_pt_bins = ak.Array([TXbb_pt_corr_bins[wp] for wp in TXbb_pt_corr_bins])
        txbb_range = [ak.min(all_txbb_bins), ak.max(all_txbb_bins)]
        pt_range = [ak.min(all_pt_bins), ak.max(all_pt_bins)]
        for ijet in get_jets_for_txbb_sf(key):
            txbb_sf_weight *= corrections.restrict_SF(
                txbb_sf["nominal"],
                bdt_events[f"H{ijet}TXbb"].to_numpy(),
                bdt_events[f"H{ijet}Pt"].to_numpy(),
                txbb_range,
                pt_range,
            )

        # remove training events if asked
        if (
            args.training_years is not None
            and year in args.training_years
            and key in bdt_training_keys
        ):
            bdt_events["event"] = events_dict["event"][0]
            inferences_dir = Path(
                f"../boosted/bdt_trainings_run3/{args.bdt_model}/inferences/{year}"
            )

            evt_list = np.load(inferences_dir / f"evt_{key}.npy")
            events_to_keep = bdt_events["event"].isin(evt_list)
            fraction = np.sum(events_to_keep.to_numpy() == 1) / bdt_events["event"].shape[0]
            logger.info(f"Keep {fraction}% of {key} for year {year}")
            bdt_events = bdt_events[events_to_keep]
            bdt_events["weight"] *= 1 / fraction  # divide by BDT test / train ratio

        cutflow_dict[key] = OrderedDict([("Skimmer Preselection", np.sum(bdt_events["weight"]))])

        # tt corrections
        ttbar_weight = np.ones(nevents)
        if args.vbf:
            mask_vbf = (bdt_events["bdt_score_vbf"] > args.vbf_bdt_wp) & (
                bdt_events["H2TXbb"] > args.vbf_txbb_wp
            )
            if not args.vbf_priority:
                mask_bin1 = (bdt_events["H2TXbb"] > args.txbb_wps[0]) & (
                    bdt_events["bdt_score"] > args.bdt_wps[0]
                )
                # prioritize bin 1 i.e. veto events in VBF region that pass the bin 1 selection
                mask_vbf = mask_vbf & ~(mask_bin1)
        else:
            # if no VBF region, set all events to "fail VBF"
            mask_vbf = np.zeros(len(bdt_events), dtype=bool)

        if key == "ttbar":
            ptjjsf, _, _ = corrections.ttbar_SF(tt_ptjj_sf, bdt_events, "HHPt")
            tau32j1sf, tau32j1sf_up, tau32j1sf_dn = corrections.ttbar_SF(
                tt_tau32_sf, bdt_events, "H1T32"
            )
            tau32j2sf, tau32j2sf_up, tau32j2sf_dn = corrections.ttbar_SF(
                tt_tau32_sf, bdt_events, "H2T32"
            )
            tau32sf = tau32j1sf * tau32j2sf
            tau32sf_up = tau32j1sf_up * tau32j2sf_up
            tau32sf_dn = tau32j1sf_dn * tau32j2sf_dn

            # inclusive xbb correction
            tempw1, _, _ = corrections.ttbar_SF(tt_xbb_sf, bdt_events, "H1TXbb")
            tempw2, _, _ = corrections.ttbar_SF(tt_xbb_sf, bdt_events, "H2TXbb")
            txbbsf = tempw1 * tempw2

            # inclusive bdt shape correction
            ggfbdtsf, _, _ = corrections.ttbar_SF(tt_ggfbdtshape_sf, bdt_events, "bdt_score")
            bdtsf = ggfbdtsf
            # use bdt_vbf correction for vbf category if it exists
            if args.correct_vbf_bdt_shape:
                vbfbdtsf, _, _ = corrections.ttbar_SF(
                    tt_vbfbdtshape_sf, bdt_events, "bdt_score_vbf"
                )
                bdtsf[mask_vbf] = vbfbdtsf[mask_vbf]

            # total ttbar correction
            ttbar_weight = ptjjsf * tau32sf * txbbsf * bdtsf

        # save corrected weights
        weights_to_correct = (
            ["weight"]
            + [f"scale_weights_{i}" for i in range(6)]
            + [f"pdf_weights_{i}" for i in range(n_pdf_weights)]
            + pileup_ps_weights
        )
        for w in weights_to_correct:
            if w in bdt_events:
                bdt_events[w] *= trigger_weight * ttbar_weight * txbb_sf_weight

        # uncorrelated signal xbb up/dn variations in bins
        for wp in TXbb_wps:
            for j in range(len(TXbb_pt_corr_bins[wp]) - 1):
                nominal = np.ones(nevents)
                stat_up = np.ones(nevents)
                stat_dn = np.ones(nevents)
                for ijet in get_jets_for_txbb_sf(key):
                    nominal *= corrections.restrict_SF(
                        txbb_sf["nominal"],
                        bdt_events[f"H{ijet}TXbb"].to_numpy(),
                        bdt_events[f"H{ijet}Pt"].to_numpy(),
                        TXbb_wps[wp],
                        TXbb_pt_corr_bins[wp][j : j + 2],
                    )
                    stat_up *= corrections.restrict_SF(
                        txbb_sf["stat_up"],
                        bdt_events[f"H{ijet}TXbb"].to_numpy(),
                        bdt_events[f"H{ijet}Pt"].to_numpy(),
                        TXbb_wps[wp],
                        TXbb_pt_corr_bins[wp][j : j + 2],
                    )
                    stat_dn *= corrections.restrict_SF(
                        txbb_sf["stat_dn"],
                        bdt_events[f"H{ijet}TXbb"].to_numpy(),
                        bdt_events[f"H{ijet}Pt"].to_numpy(),
                        TXbb_wps[wp],
                        TXbb_pt_corr_bins[wp][j : j + 2],
                    )
                bdt_events[
                    f"weight_TXbbSF_uncorrelated_{wp}_pT_bin_{TXbb_pt_corr_bins[wp][j]}_{TXbb_pt_corr_bins[wp][j+1]}Up"
                ] = (bdt_events["weight"] * stat_up / nominal)
                bdt_events[
                    f"weight_TXbbSF_uncorrelated_{wp}_pT_bin_{TXbb_pt_corr_bins[wp][j]}_{TXbb_pt_corr_bins[wp][j+1]}Down"
                ] = (bdt_events["weight"] * stat_dn / nominal)

        if key == "ttbar":
            # ttbar xbb up/dn variations in bins'
            for i in range(len(ttsf_xbb_bins) - 1):
                tempw1, tempw1_up, tempw1_dn = corrections.ttbar_SF(
                    tt_xbb_sf, bdt_events, "H1TXbb", ttsf_xbb_bins[i : i + 2]
                )
                tempw2, tempw2_up, tempw2_dn = corrections.ttbar_SF(
                    tt_xbb_sf, bdt_events, "H2TXbb", ttsf_xbb_bins[i : i + 2]
                )
                bdt_events[f"weight_ttbarSF_Xbb_bin_{ttsf_xbb_bins[i]}_{ttsf_xbb_bins[i+1]}Up"] = (
                    bdt_events["weight"] * tempw1_up * tempw2_up / (tempw1 * tempw2)
                )
                bdt_events[
                    f"weight_ttbarSF_Xbb_bin_{ttsf_xbb_bins[i]}_{ttsf_xbb_bins[i+1]}Down"
                ] = (bdt_events["weight"] * tempw1_dn * tempw2_dn / (tempw1 * tempw2))

            # bdt up/dn variations in bins
            for i in range(len(ttsf_ggfbdtshape_bins) - 1):
                ggfbdtsf, ggfbdtsf_up, ggfbdtsf_dn = corrections.ttbar_SF(
                    tt_ggfbdtshape_sf,
                    bdt_events,
                    "bdt_score",
                    ttsf_ggfbdtshape_bins[i : i + 2],
                )
                if args.correct_vbf_bdt_shape:
                    # only use ggf correction/uncertainty outside of vbf category
                    ggfbdtsf[mask_vbf] = np.ones(np.sum(mask_vbf))
                    ggfbdtsf_up[mask_vbf] = np.ones(np.sum(mask_vbf))
                    ggfbdtsf_dn[mask_vbf] = np.ones(np.sum(mask_vbf))
                bdt_events[
                    f"weight_ttbarSF_ggF_BDT_bin_{ttsf_ggfbdtshape_bins[i]}_{ttsf_ggfbdtshape_bins[i+1]}Up"
                ] = (bdt_events["weight"] * ggfbdtsf_up / ggfbdtsf)
                bdt_events[
                    f"weight_ttbarSF_ggF_BDT_bin_{ttsf_ggfbdtshape_bins[i]}_{ttsf_ggfbdtshape_bins[i+1]}Down"
                ] = (bdt_events["weight"] * ggfbdtsf_dn / ggfbdtsf)
            if args.correct_vbf_bdt_shape:
                for i in range(len(ttsf_vbfbdtshape_bins) - 1):
                    vbfbdtsf, vbfbdtsf_up, vbfbdtsf_dn = corrections.ttbar_SF(
                        tt_vbfbdtshape_sf,
                        bdt_events,
                        "bdt_score_vbf",
                        ttsf_vbfbdtshape_bins[i : i + 2],
                    )
                    # only use vbf correction/uncertainty inside of vbf category
                    vbfbdtsf[~mask_vbf] = np.ones(np.sum(~mask_vbf))
                    vbfbdtsf_up[~mask_vbf] = np.ones(np.sum(~mask_vbf))
                    vbfbdtsf_dn[~mask_vbf] = np.ones(np.sum(~mask_vbf))
                    bdt_events[
                        f"weight_ttbarSF_VBF_BDT_bin_{ttsf_vbfbdtshape_bins[i]}_{ttsf_vbfbdtshape_bins[i+1]}Up"
                    ] = (bdt_events["weight"] * vbfbdtsf_up / vbfbdtsf)
                    bdt_events[
                        f"weight_ttbarSF_VBF_BDT_bin_{ttsf_vbfbdtshape_bins[i]}_{ttsf_vbfbdtshape_bins[i+1]}Down"
                    ] = (bdt_events["weight"] * vbfbdtsf_dn / vbfbdtsf)

        if key != "data":
            bdt_events["weight_triggerUp"] = (
                bdt_events["weight"] * trigger_weight_up / trigger_weight
            )
            bdt_events["weight_triggerDown"] = (
                bdt_events["weight"] * trigger_weight_dn / trigger_weight
            )
        if key == "ttbar":
            bdt_events["weight_ttbarSF_pTjjUp"] = bdt_events["weight"] * ptjjsf
            bdt_events["weight_ttbarSF_pTjjDown"] = bdt_events["weight"] / ptjjsf
            bdt_events["weight_ttbarSF_tau32Up"] = bdt_events["weight"] * tau32sf_up / tau32sf
            bdt_events["weight_ttbarSF_tau32Down"] = bdt_events["weight"] * tau32sf_dn / tau32sf

        # HLT selection
        mask_hlt = bdt_events["hlt"] == 1
        bdt_events = bdt_events[mask_hlt]
        cutflow_dict[key]["HLT"] = np.sum(bdt_events["weight"].to_numpy())

        if args.txbb == "pnet-legacy":
            txbb_presel = 0.8
        elif args.txbb in ["glopart-v2", "pnet-v12"]:
            txbb_presel = 0.3

        for jshift in jshifts:
            logger.info(f"Inference and selection for jshift {jshift}")
            h1pt = check_get_jec_var("H1Pt", jshift)
            h2pt = check_get_jec_var("H2Pt", jshift)
            h1msd = check_get_jec_var("H1Msd", jshift)
            h1mass = check_get_jec_var(args.mass.replace("H2", "H1"), jshift)
            h2mass = check_get_jec_var(args.mass, jshift)
            category = check_get_jec_var("Category", jshift)
            bdt_score = check_get_jec_var("bdt_score", jshift)

            mask_presel = (
                (bdt_events[h1msd] >= 40)  # FIXME: replace by jet matched to trigger object
                & (bdt_events[h1pt] >= args.pt_first)
                & (bdt_events[h2pt] >= args.pt_second)
                & (bdt_events["H1TXbb"] >= txbb_presel)
                & (bdt_events[h2mass] >= 60)
                & (bdt_events[h2mass] <= 220)
                & (bdt_events[h1mass] >= 60)
                & (bdt_events[h1mass] <= 220)
            )
            bdt_events = bdt_events[mask_presel]

            ###### FINISH pre-selection
            mass_str = f"[{mass_window[0]}-{mass_window[1]}]"
            mask_mass = (bdt_events[h2mass] >= mass_window[0]) & (
                bdt_events[h2mass] <= mass_window[1]
            )

            # define category
            bdt_events[category] = 5  # all events

            mask_fail = (bdt_events["H2TXbb"] < args.txbb_wps[1]) & (
                bdt_events[bdt_score] > args.bdt_wps[2]
            )
            bdt_events.loc[mask_fail, category] = 4

            if args.vbf:
                bdt_score_vbf = check_get_jec_var("bdt_score_vbf", jshift)
                mask_vbf = (bdt_events[bdt_score_vbf] > args.vbf_bdt_wp) & (
                    bdt_events["H2TXbb"] > args.vbf_txbb_wp
                )
            else:
                # if no VBF region, set all events to "fail VBF"
                mask_vbf = np.zeros(len(bdt_events), dtype=bool)

            mask_bin1 = (bdt_events["H2TXbb"] > args.txbb_wps[0]) & (
                bdt_events[bdt_score] > args.bdt_wps[0]
            )

            if args.vbf_priority:
                # prioritize VBF region i.e. veto events in bin1 that pass the VBF selection
                mask_bin1 = mask_bin1 & ~(mask_vbf)
            else:
                # prioritize bin 1 i.e. veto events in VBF region that pass the bin 1 selection
                mask_vbf = mask_vbf & ~(mask_bin1)

            bdt_events.loc[mask_vbf, category] = 0

            bdt_events.loc[mask_bin1, category] = 1

            mask_corner = (bdt_events["H2TXbb"] < args.txbb_wps[0]) & (
                bdt_events[bdt_score] < args.bdt_wps[0]
            )
            mask_bin2 = (
                (bdt_events["H2TXbb"] > args.txbb_wps[1])
                & (bdt_events[bdt_score] > args.bdt_wps[1])
                & ~(mask_bin1)
                & ~(mask_corner)
                & ~(mask_vbf)
            )
            bdt_events.loc[mask_bin2, category] = 2

            mask_bin3 = (
                (bdt_events["H2TXbb"] > args.txbb_wps[1])
                & (bdt_events[bdt_score] > args.bdt_wps[2])
                & ~(mask_bin1)
                & ~(mask_bin2)
                & ~(mask_vbf)
            )
            bdt_events.loc[mask_bin3, category] = 3

        # save cutflows for nominal variables
        cutflow_dict[key][f"H1Msd > 40 & H2Pt > {args.pt_second} & H1Pt > {args.pt_first}"] = (
            np.sum(bdt_events["weight"].to_numpy())
        )

        cutflow_dict[key]["BDT > min"] = np.sum(
            bdt_events["weight"][bdt_events["bdt_score"] > args.bdt_wps[2]].to_numpy()
        )

        cutflow_dict[key][f"Bin VBF {mass_str}"] = np.sum(
            bdt_events["weight"][mask_vbf & mask_mass].to_numpy()
        )
        cutflow_dict[key]["Bin VBF"] = np.sum(bdt_events["weight"][mask_vbf].to_numpy())
        cutflow_dict[key][f"Bin VBF {mass_str}"] = np.sum(
            bdt_events["weight"][mask_vbf & mask_mass].to_numpy()
        )

        cutflow_dict[key]["Bin 1"] = np.sum(bdt_events["weight"][mask_bin1].to_numpy())
        cutflow_dict[key][f"Bin 1 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin1 & mask_mass].to_numpy()
        )

        cutflow_dict[key]["VBF & Bin 1 overlap"] = np.sum(
            bdt_events["weight"][
                (bdt_events["H2TXbb"] > args.txbb_wps[0])
                & (bdt_events["bdt_score"] > args.bdt_wps[0])
                & mask_vbf
            ].to_numpy()
        )

        cutflow_dict[key]["Bin 2"] = np.sum(bdt_events["weight"][mask_bin2].to_numpy())
        cutflow_dict[key][f"Bin 2 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin2 & mask_mass].to_numpy()
        )

        cutflow_dict[key]["Bin 3"] = np.sum(bdt_events["weight"][mask_bin3].to_numpy())
        cutflow_dict[key][f"Bin 3 {mass_str}"] = np.sum(
            bdt_events["weight"][mask_bin3 & mask_mass].to_numpy()
        )

        # save year as column
        bdt_events["year"] = year

        # keep some (or all) columns
        columns = [
            "Category",
            "H2Msd",
            "bdt_score",
            "H2TXbb",
            "H2PNetMass",
            "weight",
            "event",
            "run",
            "luminosityBlock",
            "year",
        ]
        for jshift in jshifts:
            columns += [
                check_get_jec_var("Category", jshift),
                check_get_jec_var("bdt_score", jshift),
                check_get_jec_var("H2Msd", jshift),
                check_get_jec_var("H2PNetMass", jshift),
            ]
        if "bdt_score_vbf" in bdt_events:
            columns += [check_get_jec_var("bdt_score_vbf", jshift) for jshift in jshifts]
        if key in hh_vars.sig_keys + ["ttbar"]:
            for i in range(6):
                columns += [f"scale_weights_{i}"]
        if key == "ttbar":
            columns += [column for column in bdt_events.columns if "weight_ttbarSF" in column]
        if key in hh_vars.sig_keys + [
            "vhtobb",
            "zz",
            "novhhtobb",
            "tthtobb",
            "vjets",
            "nozzdiboson",
        ]:
            columns += [column for column in bdt_events.columns if "weight_TXbbSF" in column]
        if key in hh_vars.sig_keys:
            for i in range(n_pdf_weights):
                columns += [f"pdf_weights_{i}"]
        if key != "data":
            columns += ["weight_triggerUp", "weight_triggerDown"] + pileup_ps_weights
        columns = list(set(columns))

        if control_plots:
            bdt_events = bdt_events.rename(
                columns={
                    "H1T32": "H1T32top",
                    "H2T32": "H2T32top",
                    "H1Pt/H2Pt": "H1Pt_H2Pt",
                    "H1AK4JetAway1dR": "H1dRAK4r",
                    "H2AK4JetAway2dR": "H2dRAK4r",
                    "H1AK4JetAway1mass": "H1AK4mass",
                    "H2AK4JetAway2mass": "H2AK4mass",
                },
            )
            events_dict_postprocess[key] = bdt_events
            columns_by_key[key] = columns
        else:
            events_dict_postprocess[key] = bdt_events[columns]

        # blind!!
        if key == "data":
            # get sideband estimate instead
            logger.info(f"Data cutflow in {mass_str} is taken from sideband estimate!")
            cutflow_dict[key][f"Bin VBF {mass_str}"] = get_nevents_data(
                bdt_events, mask_vbf, args.mass, mass_window
            )
            cutflow_dict[key][f"Bin 1 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin1, args.mass, mass_window
            )
            cutflow_dict[key][f"Bin 2 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin2, args.mass, mass_window
            )
            cutflow_dict[key][f"Bin 3 {mass_str}"] = get_nevents_data(
                bdt_events, mask_bin3, args.mass, mass_window
            )
    # end of loop over samples
    """
    if control_plots:
        make_control_plots(events_dict_postprocess, plot_dir, year, args.txbb)
        for key in events_dict_postprocess:
            events_dict_postprocess[key] = events_dict_postprocess[key][columns_by_key[key]]
    """

    """
    if "hh4b" in cutflow_dict:
        for cut in cutflow_dict["hh4b"]:
            cutflow[cut] = [
                cutflow_dict[key][cut].round(4) if cut in cutflow_dict[key] else -1.0
                for key in events_dict_postprocess
            ]

    logger.info(f"\nCutflow {cutflow}")
    """
    return events_dict_postprocess, cutflow


def get_nevents_data(events, cut, mass, mass_window):
    mw_size = mass_window[1] - mass_window[0]

    # get yield in left sideband
    cut_mass_0 = (events[mass] < mass_window[0]) & (events[mass] > (mass_window[0] - mw_size / 2))

    # get yield in right sideband
    cut_mass_1 = (events[mass] < mass_window[1] + mw_size / 2) & (events[mass] > mass_window[1])

    return np.sum((cut_mass_0 | cut_mass_1) & cut)


def get_nevents_signal(events, cut, mass, mass_window):
    cut_mass = (events[mass] >= mass_window[0]) & (events[mass] <= mass_window[1])

    # get yield in Higgs mass window
    return np.sum(events["weight"][cut & cut_mass])


def get_nevents_nosignal(events, cut, mass, mass_window):
    cut_mass = ((events[mass] >= 60) & (events[mass] <= mass_window[0])) | (
        (events[mass] >= mass_window[1]) & (events[mass] <= 220)
    )

    # get yield NOT in Higgs mass window
    return np.sum(events["weight"][cut & cut_mass])


def scan_fom(
    method: str,
    events_combined: pd.DataFrame,
    get_cut: Callable,
    get_anti_cut: Callable,
    xbb_cuts: np.ArrayLike,
    bdt_cuts: np.ArrayLike,
    mass_window: list[float],
    plot_dir: str,
    plot_name: str,
    bg_keys: list[str],
    sig_keys: list[str],
    fom: str = "2sqrt(b)/s",
    mass: str = "H2Msd",
):
    """Generic FoM scan for given region, defined in the ``get_cut`` function."""
    h_sb = hist.Hist(
        hist.axis.Variable(list(bdt_cuts) + [1.0], name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts) + [1.0], name="xbb_cut"),
    )

    h_b = hist.Hist(
        hist.axis.Variable(list(bdt_cuts) + [1.0], name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts) + [1.0], name="xbb_cut"),
    )

    h_b_unc = hist.Hist(
        hist.axis.Variable(list(bdt_cuts) + [1.0], name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts) + [1.0], name="xbb_cut"),
    )

    h_sideband = hist.Hist(
        hist.axis.Variable(list(bdt_cuts) + [1.0], name="bdt_cut"),
        hist.axis.Variable(list(xbb_cuts) + [1.0], name="xbb_cut"),
    )

    print(f"Scanning {fom} with {method}")
    all_s = []
    all_b = []
    all_b_unc = []
    all_sideband_events = []
    all_xbb_cuts = []
    all_bdt_cuts = []
    all_fom = []
    for xbb_cut in xbb_cuts:
        for bdt_cut in bdt_cuts:
            if method == "abcd":
                nevents_sig, nevents_bkg, _ = abcd(
                    events_combined,
                    get_cut,
                    get_anti_cut,
                    xbb_cut,
                    bdt_cut,
                    mass,
                    mass_window,
                    bg_keys,
                    sig_keys,
                )
            else:
                nevents_sig, nevents_bkg, _ = sideband(
                    events_combined, get_cut, xbb_cut, bdt_cut, mass, mass_window, sig_keys
                )

            # number of events in data in sideband
            cut = get_cut(events_combined["data"], xbb_cut, bdt_cut)
            nevents_sideband = get_nevents_nosignal(events_combined["data"], cut, mass, mass_window)

            if fom == "s/sqrt(s+b)":
                figure_of_merit = nevents_sig / np.sqrt(nevents_sig + nevents_bkg)
            elif fom == "2sqrt(b)/s":
                figure_of_merit = 2 * np.sqrt(nevents_bkg) / nevents_sig
            else:
                raise ValueError("Invalid FOM")

            # if nevents_sig > 0.5 and nevents_bkg >= 2 and nevents_sideband >= 12:
            # save all cuts for finetuning constraint after
            h_sb.fill(bdt_cut, xbb_cut, weight=figure_of_merit)
            h_b.fill(bdt_cut, xbb_cut, weight=nevents_bkg)
            h_b_unc.fill(bdt_cut, xbb_cut, weight=np.sqrt(nevents_bkg))
            h_sideband.fill(bdt_cut, xbb_cut, weight=nevents_sideband)
            all_b.append(nevents_bkg)
            all_b_unc.append(np.sqrt(nevents_bkg))
            all_s.append(nevents_sig)
            all_sideband_events.append(nevents_sideband)
            all_xbb_cuts.append(xbb_cut)
            all_bdt_cuts.append(bdt_cut)
            all_fom.append(figure_of_merit)

    all_fom = np.array(all_fom)
    all_b = np.array(all_b)
    all_b_unc = np.array(all_b_unc)
    all_s = np.array(all_s)
    all_sideband_events = np.array(all_sideband_events)
    all_xbb_cuts = np.array(all_xbb_cuts)
    all_bdt_cuts = np.array(all_bdt_cuts)

    # save all arrays to plot_dir
    name = f"{plot_name}_{args.method}_mass{mass_window[0]}-{mass_window[1]}"
    print(f"Saving FOM scan: {plot_dir}/{name}_fom_arrays \n")
    np.savez(
        f"{plot_dir}/{name}_fom_arrays.npz",
        all_fom=all_fom,
        all_b=all_b,
        all_b_unc=all_b_unc,
        all_s=all_s,
        all_sideband_events=all_sideband_events,
        all_xbb_cuts=all_xbb_cuts,
        all_bdt_cuts=all_bdt_cuts,
    )

    # plot fom scan
    print(f"Plotting FOM scan: {plot_dir}/{name} \n")
    plotting.plot_fom(h_sb, plot_dir, name=name, fontsize=2.0)
    plotting.plot_fom(h_b, plot_dir, name=f"{name}_bkg", fontsize=2.0)
    plotting.plot_fom(h_b_unc, plot_dir, name=f"{name}_bkgunc", fontsize=2.0)
    plotting.plot_fom(h_sideband, plot_dir, name=f"{name}_sideband", fontsize=2.0)


def get_anti_cuts(args, region: str):

    def anti_cut_vbf(events):
        cut_xbb = events["H2TXbb"] < 0.8 if args.txbb == "pnet-legacy" else events["H2TXbb"] < 0.3
        cut_bdt = events["bdt_score_vbf"] < 0.6
        return cut_xbb & cut_bdt

    def anti_cut_ggf(events):
        cut_xbb = events["H2TXbb"] < 0.8 if args.txbb == "pnet-legacy" else events["H2TXbb"] < 0.3
        cut_bdt = events["bdt_score"] < 0.6
        return cut_xbb & cut_bdt

    if region == "vbf":
        return anti_cut_vbf
    else:
        return anti_cut_ggf


def get_cuts(args, region: str):
    xbb_cut_bin1 = args.txbb_wps[0]
    bdt_cut_bin1 = args.bdt_wps[0]

    # VBF region
    def get_cut_vbf(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score_vbf"] > bdt_cut
        return cut_xbb & cut_bdt

    def get_cut_novbf(events, xbb_cut, bdt_cut):  # noqa: ARG001
        return np.zeros(len(events), dtype=bool)

    # VBF with bin1 veto
    def get_cut_vbf_vetobin1(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2TXbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score_vbf"] > bdt_cut
        return cut_xbb & cut_bdt & (~cut_bin1)

    # bin 1 with VBF region veto
    def get_cut_bin1_vetovbf(events, xbb_cut, bdt_cut):
        vbf_cut = (events["bdt_score_vbf"] >= args.vbf_bdt_wp) & (
            events["H2TXbb"] >= args.vbf_txbb_wp
        )
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        return cut_xbb & cut_bdt & (~vbf_cut)

    # bin 1 region
    def get_cut_bin1(events, xbb_cut, bdt_cut):
        cut_xbb = events["H2TXbb"] > xbb_cut
        cut_bdt = events["bdt_score"] > bdt_cut
        return cut_xbb & cut_bdt

    # bin 2 with VBF region veto
    def get_cut_bin2_vetovbf(events, xbb_cut, bdt_cut):
        vbf_cut = (events["bdt_score_vbf"] >= args.vbf_bdt_wp) & (
            events["H2TXbb"] >= args.vbf_txbb_wp
        )
        cut_bin1 = (events["H2TXbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2TXbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2TXbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
            & ~(vbf_cut)
        )

        return cut_bin2

    # bin 2 without VBF region veto
    def get_cut_bin2(events, xbb_cut, bdt_cut):
        cut_bin1 = (events["H2TXbb"] > xbb_cut_bin1) & (events["bdt_score"] > bdt_cut_bin1)
        cut_corner = (events["H2TXbb"] < xbb_cut_bin1) & (events["bdt_score"] < bdt_cut_bin1)
        cut_bin2 = (
            (events["H2TXbb"] > xbb_cut)
            & (events["bdt_score"] > bdt_cut)
            & ~(cut_bin1)
            & ~(cut_corner)
        )

        return cut_bin2

    if region == "vbf":
        if args.vbf and args.vbf_priority:
            return get_cut_vbf
        elif args.vbf and not args.vbf_priority:
            return get_cut_vbf_vetobin1
        else:
            # if no VBF region, set all events to "fail VBF"
            return get_cut_novbf
    elif region == "bin1":
        return get_cut_bin1_vetovbf if (args.vbf and args.vbf_priority) else get_cut_bin1
    elif region == "bin2":
        return get_cut_bin2_vetovbf if args.vbf else get_cut_bin2
    else:
        raise ValueError("Invalid region")


def make_control_plots(events_dict, plot_dir, year, txbb_version):

    if txbb_version == "pnet-legacy":
        txbb_label = "PNet Legacy"
    elif txbb_version == "pnet-v12":
        txbb_label = "PNet 103X"
    elif txbb_version == "glopart-v2":
        txbb_label = "GloParTv2"

    control_plot_vars = [
        ShapeVar(var="bdt_score", label=r"BDT score ggF", bins=[30, 0, 1], blind_window=[0.8, 1.0]),
        ShapeVar(
            var="bdt_score_vbf", label=r"BDT score VBF", bins=[30, 0, 1], blind_window=[0.8, 1.0]
        ),
        ShapeVar(var="H1Msd", label=r"$m_{SD}^{1}$ (GeV)", bins=[30, 0, 300]),
        ShapeVar(var="H2Msd", label=r"$m_{SD}^{2}$ (GeV)", bins=[30, 0, 300]),
        ShapeVar(var="H1TXbb", label=r"Xbb$^{1}$ " + txbb_label, bins=[30, 0, 1]),
        ShapeVar(var="H1DiscTXbb", label=r"Discretized Xbb$^{1}$ " + txbb_label, bins=[6, 1, 7]),
        ShapeVar(var="H2TXbb", label=r"Xbb$^{2}$ " + txbb_label, bins=[30, 0, 1]),
        ShapeVar(var="H2DiscTXbb", label=r"Discretized Xbb$^{2}$ " + txbb_label, bins=[6, 1, 7]),
        ShapeVar(var="H1PNetMass", label=r"$m_{reg}^{1}$ (GeV) " + txbb_label, bins=[30, 0, 300]),
        ShapeVar(var="H2PNetMass", label=r"$m_{reg}^{2}$ (GeV) " + txbb_label, bins=[30, 0, 300]),
        ShapeVar(var="HHPt", label=r"HH $p_{T}$ (GeV)", bins=[30, 0, 4000]),
        ShapeVar(var="HHeta", label=r"HH $\eta$", bins=[30, -5, 5]),
        ShapeVar(var="HHmass", label=r"HH mass (GeV)", bins=[30, 0, 1500]),
        ShapeVar(var="MET", label=r"MET (GeV)", bins=[30, 0, 600]),
        ShapeVar(var="H1T32top", label=r"$\tau_{32}^{1}$", bins=[30, 0, 1]),
        ShapeVar(var="H2T32top", label=r"$\tau_{32}^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1Pt", label=r"H $p_{T}^{1}$ (GeV)", bins=[30, 200, 1000]),
        ShapeVar(var="H2Pt", label=r"H $p_{T}^{2}$ (GeV)", bins=[30, 200, 1000]),
        ShapeVar(var="H1eta", label=r"H $\eta^{1}$", bins=[30, -4, 4]),
        ShapeVar(var="H1Pt_HHmass", label=r"H$^1$ $p_{T}/mass$", bins=[30, 0, 1]),
        ShapeVar(var="H2Pt_HHmass", label=r"H$^2$ $p_{T}/mass$", bins=[30, 0, 0.7]),
        ShapeVar(var="H1Pt_H2Pt", label=r"H$^1$/H$^2$ $p_{T}$ (GeV)", bins=[30, 0.5, 1]),
        ShapeVar(var="VBFjjMass", label=r"VBF jj mass (GeV)", bins=[30, 0.0, 1000]),
        ShapeVar(var="VBFjjDeltaEta", label=r"VBF jj $\Delta \eta$", bins=[30, 0, 5]),
        ShapeVar(var="H1dRAK4r", label=r"$\Delta R$(H1,J1)", bins=[30, 0, 5]),
        ShapeVar(var="H2dRAK4r", label=r"$\Delta R$(H2,J2)", bins=[30, 0, 5]),
        ShapeVar(var="H1AK4mass", label=r"(H1 + J1) mass (GeV)", bins=[30, 80, 600]),
        ShapeVar(var="H2AK4mass", label=r"(H2 + J2) mass (GeV)", bins=[30, 80, 600]),
        # these are not used for BDT
        ShapeVar(var="H1Msd", label=r"$m_{SD}^{1}$ (GeV)", bins=[40, 0, 300]),
        ShapeVar(var="H2Msd", label=r"$m_{SD}^{2}$ (GeV)", bins=[30, 0, 300]),
        ShapeVar(var="H1QCDb", label=r"QCDb$^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1QCDbb", label=r"QCDbb$^{2}$", bins=[30, 0, 1]),
        ShapeVar(var="H1QCDothers", label=r"QCDothers$^{1}$", bins=[30, 0, 1]),
    ]

    (plot_dir / f"control/{year}").mkdir(exist_ok=True, parents=True)

    # Find the normalization needed to reweight QCD
    qcd_norm = 1.0

    hists = {}
    for i, shape_var in enumerate(control_plot_vars):
        if shape_var.var not in hists:
            hists[shape_var.var] = singleVarHist(
                events_dict,
                shape_var,
                weight_key="weight",
            )

            qcd_norm_tmp = plotting.ratioHistPlot(
                hists[shape_var.var],
                year,
                ["hh4b", "vbfhh4b", "vbfhh4b-k2v0"],
                bg_keys,
                name=f"{plot_dir}/control/{year}/{shape_var.var}",
                show=False,
                log=True,
                plot_significance=False,
                significance_dir=shape_var.significance_dir,
                ratio_ylims=[0.2, 1.8],
                bg_err_mcstat=True,
                reweight_qcd=True,
                qcd_norm=qcd_norm if i != 0 else None,
            )

            # pick the normalization weight chosen for the first variable
            qcd_norm = qcd_norm_tmp


def sideband(events_dict, get_cut, txbb_cut, bdt_cut, mass, mass_window, sig_keys):
    nevents_bkg = get_nevents_data(
        events_dict["data"],
        get_cut(events_dict["data"], txbb_cut, bdt_cut),
        mass,
        mass_window,
    )
    nevents_sig = 0
    for sig_key in sig_keys:
        nevents_sig += get_nevents_signal(
            events_dict[sig_key],
            get_cut(events_dict[sig_key], txbb_cut, bdt_cut),
            mass,
            mass_window,
        )
    return nevents_sig, nevents_bkg, {}


def abcd(
    events_dict,
    get_cut,
    get_anti_cut,
    txbb_cut,
    bdt_cut,
    mass,
    mass_window,
    bg_keys_all,
    sig_keys,
):
    bg_keys = bg_keys_all.copy()
    if "qcd" in bg_keys:
        bg_keys.remove("qcd")

    dicts = {"data": [], **{key: [] for key in bg_keys}}

    s = 0
    for key in sig_keys + ["data"] + bg_keys:
        events = events_dict[key]
        cut = get_cut(events, txbb_cut, bdt_cut)

        if key in sig_keys:
            s += get_nevents_signal(events, cut, mass, mass_window)
            continue

        # region A
        if key == "data":
            dicts[key].append(0)
        else:
            dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))

        # region B
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

        cut = get_anti_cut(events)

        # region C
        dicts[key].append(get_nevents_signal(events, cut, mass, mass_window))
        # region D
        dicts[key].append(get_nevents_nosignal(events, cut, mass, mass_window))

    # other backgrounds (bg_tots[0] is in region A)
    bg_tots = np.sum([dicts[key] for key in bg_keys], axis=0)
    # subtract other backgrounds
    dmt = np.array(dicts["data"]) - bg_tots
    # A = B * C / D
    bqcd = dmt[1] * dmt[2] / dmt[3]

    background = bqcd + bg_tots[0] if len(bg_keys) else bqcd
    return s, background, dicts


def postprocess_run3(args):
    global bg_keys  # noqa: PLW0602

    fom_window_by_mass = {
        "H2Msd": [110, 140],
    }
    blind_window_by_mass = {
        "H2Msd": [110, 140],
    }

    # use for both pnet-legacy
    if args.txbb == "pnet-legacy":
        fom_window_by_mass["H2PNetMass"] = [105, 150]  # use wider range for FoM scan
        blind_window_by_mass["H2PNetMass"] = [110, 140]  # only blind 3 bins
    # different for glopart-v2
    elif args.txbb == "glopart-v2":
        fom_window_by_mass["H2PNetMass"] = [110, 155]  # use wider range for FoM scan
        blind_window_by_mass["H2PNetMass"] = [110, 140]  # only blind 3 bins
    # different for pnet-v12
    elif args.txbb == "pnet-v12":
        fom_window_by_mass["H2PNetMass"] = [120, 150]
        blind_window_by_mass["H2PNetMass"] = [120, 150]

    mass_window = np.array(fom_window_by_mass[args.mass])

    n_mass_bins = int((220 - 60) / args.mass_bins)

    # variable to fit
    fit_shape_var = ShapeVar(
        args.mass,
        label_by_mass[args.mass],
        [n_mass_bins, 60, 220],
        reg=True,
        blind_window=blind_window_by_mass[args.mass],
    )

    # add weight shifts based on xbb and bdt versions
    weight_shifts = get_weight_shifts(args.txbb, args.bdt_model)

    plot_dir = Path(f"{HH4B_DIR}/plots/PostProcess/{args.templates_tag}")
    plot_dir.mkdir(exist_ok=True, parents=True)

    # load samples
    try:
        bdt_training_keys = get_bdt_training_keys(args.bdt_model)
    except FileNotFoundError:
        print("File with training events is not available")
        bdt_training_keys = []
    events_dict_postprocess = {}
    cutflows = {}
    for year in args.years:
        print(f"\n{year}")
        events, cutflow = load_process_run3_samples(
            args,
            year,
            bdt_training_keys,
            args.control_plots,
            plot_dir,
            mass_window,
            args.rerun_inference,
        )
        events_dict_postprocess[year] = events
        cutflows[year] = cutflow

    print("Loaded all years")

    processes = ["data"] + args.sig_keys + bg_keys
    bg_keys_combined = bg_keys.copy()
    if not args.control_plots and not args.bdt_roc:
        if "qcd" in processes:
            processes.remove("qcd")
        if "qcd" in bg_keys:
            bg_keys.remove("qcd")
        if "qcd" in bg_keys_combined:
            bg_keys_combined.remove("qcd")

    print("BKG keys ", bg_keys)

    if len(args.years) > 1:
        # list of years available for a given process to scale to full lumi
        available = [y for y in ["2022", "2022EE", "2023", "2023BPix"] if y in args.years]
        print(f"WARNING: Using available MC from {available}")
        scaled_by_years = {
            "ttbar": available,
            "novhhtobb": available,
            "vhtobb": available,
            "tthtobb": available,
            "zz": available,
            "nozzdiboson": available,
            "vjets": available,
            "qcd": available,
        }
        events_combined, scaled_by = combine_run3_samples(
            events_dict_postprocess,
            processes,
            bg_keys=bg_keys_combined,
            scale_processes=scaled_by_years,
            years_run3=args.years,
        )
        print("Combined years")
    else:
        events_combined = events_dict_postprocess[args.years[0]]
        scaled_by = {}
    if args.control_plots:
        # quick fix: '2024' is only stand-in, plots all combined events
        # uses 2022-2013 for scaled MC
        make_control_plots(events_combined, plot_dir, "2024", args.txbb)

    if args.bdt_roc:
        print("Making BDT ROC curve")
        bdt_roc(events_combined, plot_dir, args.txbb)
        # to make ROC curves for JMR variations
        # bdt_roc(events_combined, plot_dir, args.txbb, jshift="JMR_up")
        # bdt_roc(events_combined, plot_dir, args.txbb, jshift="JMR_down")

    # combined cutflow
    cutflow_combined = None
    if len(args.years) > 0:
        cutflow_combined = pd.DataFrame(index=list(events_combined.keys()))

        # get ABCD (warning!: not considering VBF region veto)
        (
            s_bin1,
            b_bin1,
            _,
        ) = abcd(
            events_combined,
            get_cuts(args, "bin1"),
            get_anti_cuts(args, "bin1"),
            args.txbb_wps[0],
            args.bdt_wps[0],
            args.mass,
            mass_window,
            bg_keys,
            ["hh4b"],
        )

        s_binVBF, b_binVBF, _ = abcd(
            events_combined,
            get_cuts(args, "vbf"),
            get_anti_cuts(args, "vbf"),
            args.txbb_wps[0],
            args.bdt_wps[0],
            args.mass,
            mass_window,
            bg_keys,
            ["hh4b"],
        )

        # note: need to do this since not all the years have all the samples..
        year_0 = "2022EE" if "2022EE" in args.years else args.years[0]
        samples = list(events_combined.keys())
        for cut in cutflows[year_0]:
            # yield_s = 0
            yield_b = 0
            for s in samples:
                if s in scaled_by:
                    cutflow_sample = 0.0
                    for year in args.years:
                        if (
                            cut in cutflows[year]
                            and s in cutflows[year][cut].index
                            and year in scaled_by_years[s]
                        ):
                            cutflow_sample += cutflows[year][cut].loc[s]
                    cutflow_sample *= scaled_by[s]
                    print(f"Scaling combined cutflow for {s} by {scaled_by[s]}")
                else:
                    cutflow_sample = np.sum(
                        [
                            (
                                cutflows[year][cut].loc[s]
                                if cut in cutflows[year] and s in cutflows[year][cut].index
                                else 0.0
                            )
                            for year in args.years
                        ]
                    )

                # if s == "hh4b":
                #    yield_s = cutflow_sample
                if s == "data":
                    yield_b = cutflow_sample
                cutflow_combined.loc[s, cut] = f"{cutflow_sample:.4f}"

            if "VBF [" in cut:
                cutflow_combined.loc["B ABCD", cut] = f"{b_binVBF:.4f}"
            if "Bin 1 [" in cut and yield_b > 0:
                cutflow_combined.loc["B ABCD", cut] = f"{b_bin1:.3f}"
                cutflow_combined.loc["S/B ABCD", cut] = f"{s_bin1/b_bin1:.3f}"

        print(f"\n Combined cutflow TXbb:{args.txbb_wps} BDT: {args.bdt_wps}")
        print(cutflow_combined)

    if args.fom_scan:
        if args.fom_scan_vbf and args.vbf:
            if args.vbf_priority:
                print("Scanning VBF WPs")
            else:
                print("Scanning VBF WPs, vetoing Bin1")
            print(f"Using bg keys {bg_keys}")
            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "vbf"),
                get_anti_cuts(args, "vbf"),
                np.arange(0.8, 0.999, 0.0025),
                np.arange(0.9, 0.999, 0.0025),
                mass_window,
                plot_dir,
                "fom_vbf",
                bg_keys=bg_keys,
                sig_keys=args.fom_vbf_samples,
                mass=args.mass,
            )

        if args.fom_scan_bin1:
            if args.vbf and args.vbf_priority:
                print(
                    f"Scanning Bin 1, vetoing VBF TXbb WP: {args.vbf_txbb_wp} BDT WP: {args.vbf_bdt_wp}"
                )
            else:
                print("Scanning Bin 1, no VBF category")

            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "bin1"),
                get_anti_cuts(args, "bin1"),
                np.arange(0.9, 0.999, 0.0025),
                np.arange(0.9, 0.999, 0.0025),
                mass_window,
                plot_dir,
                "fom_bin1",
                bg_keys=bg_keys,
                sig_keys=args.fom_ggf_samples,
                mass=args.mass,
            )

        if args.fom_scan_bin2:
            if args.vbf:
                print(
                    f"Scanning Bin 2, vetoing VBF TXbb WP: {args.vbf_txbb_wp} BDT WP: {args.vbf_bdt_wp}, bin 1 WP: {args.txbb_wps[0]} BDT WP: {args.bdt_wps[0]}"
                )
            else:
                print(
                    f"Scanning Bin 2, vetoing bin 1 WP: {args.txbb_wps[0]} BDT WP: {args.bdt_wps[0]}"
                )
            scan_fom(
                args.method,
                events_combined,
                get_cuts(args, "bin2"),
                get_anti_cuts(args, "bin2"),
                np.arange(0.5, 0.9, 0.0025),
                np.arange(0.5, 0.9, 0.0025),
                mass_window,
                plot_dir,
                "fom_bin2",
                bg_keys=bg_keys,
                sig_keys=args.fom_ggf_samples,
                mass=args.mass,
            )

    templ_dir = Path("templates") / args.templates_tag
    for year in args.years:
        (templ_dir / "cutflows" / year).mkdir(parents=True, exist_ok=True)
        (templ_dir / year).mkdir(parents=True, exist_ok=True)

    # save args for posterity
    with (templ_dir / "args.txt").open("w") as f:
        pretty_printer = pprint.PrettyPrinter(stream=f, indent=4)
        pretty_printer.pprint(vars(args))

    for cyear in args.years:
        cutflows[cyear] = cutflows[cyear].round(4)
        cutflows[cyear].to_csv(templ_dir / "cutflows" / f"preselection_cutflow_{cyear}.csv")
    if cutflow_combined is not None:
        cutflow_combined = cutflow_combined.round(4)
        cutflow_combined.to_csv(templ_dir / "cutflows" / "preselection_cutflow_combined.csv")

    if not args.templates:
        return

    if not args.vbf:
        selection_regions.pop("pass_vbf")

    # individual templates per year
    for year in args.years:
        templates = {}
        for jshift in [""] + hh_vars.jec_shifts + hh_vars.jmsr_shifts:
            events_by_year = {}
            for sample, events in events_combined.items():
                events_by_year[sample] = events[events["year"] == year]
            ttemps = postprocessing.get_templates(
                events_by_year,
                year=year,
                sig_keys=args.sig_keys,
                plot_sig_keys=["hh4b", "vbfhh4b", "vbfhh4b-k2v0"],
                selection_regions=selection_regions,
                shape_vars=[fit_shape_var],
                systematics={},
                template_dir=templ_dir,
                bg_keys=bg_keys_combined,
                plot_dir=Path(f"{templ_dir}/{year}"),
                weight_key="weight",
                weight_shifts=weight_shifts,
                plot_shifts=False,  # skip for time
                show=False,
                energy=13.6,
                jshift=jshift,
                blind=args.blind,
            )
            templates = {**templates, **ttemps}

        # save templates per year
        postprocessing.save_templates(
            templates, templ_dir / f"{year}_templates.pkl", fit_shape_var, blind=args.blind
        )
    if args.event_list:

        import uproot

        eventlist_dict = [
            "event",
            "bdt_score",
            "bdt_score_vbf",
            "H2TXbb",
            "H2Msd",
            "run",
            "Category",
            "H2PNetMass",
            "luminosityBlock",
        ]

        eventlist_folder = args.event_list_dir
        Path(eventlist_folder).mkdir(parents=True, exist_ok=True)

        for year, year_dict in events_dict_postprocess.items():
            for key, tree_df in year_dict.items():
                if "data" in key or "hh4b" in key or "vbfhh4b" in key:
                    event_list = tree_df[eventlist_dict]
                    array_to_save = {col: event_list[col].to_numpy() for col in event_list.columns}

                    # Define the ROOT file path
                    file_path = f"{eventlist_folder}/eventlist_boostedHH4b_{year}.root"

                    # Check if the ROOT file already exists
                    if Path(file_path).exists():
                        # File exists, use update mode to append the new tree
                        with uproot.update(file_path) as file:
                            file[key] = array_to_save  # Append new tree
                    else:
                        # File doesn't exist, create a new one
                        with uproot.recreate(file_path) as file:
                            file[key] = array_to_save  # Create the first tree

    # combined templates
    # skip for time
    """
    if len(args.years) > 0:
        (templ_dir / "cutflows" / "2022-2023").mkdir(parents=True, exist_ok=True)
        (templ_dir / "2022-2023").mkdir(parents=True, exist_ok=True)
        templates = postprocessing.get_templates(
            events_combined,
            year="2022-2023",
            sig_keys=args.sig_keys,
            selection_regions=selection_regions,
            shape_vars=[fit_shape_var],
            systematics={},
            template_dir=templ_dir,
            bg_keys=bg_keys_combined,
            plot_dir=Path(f"{templ_dir}/2022-2023"),
            weight_key="weight",
            weight_shifts=weight_shifts,
            plot_shifts=False,
            show=False,
            energy=13.6,
            jshift="",
        )
        postprocessing.save_templates(templates, templ_dir / "2022-2023_templates.pkl", fit_shape_var)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--templates-tag",
        type=str,
        required=True,
        help="output pickle directory of hist.Hist templates inside the ./templates dir",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/ceph/cms/store/user/cmantill/bbbb/skimmer/",
        help="tag for input ntuples",
    )
    parser.add_argument(
        "--mass-bins",
        type=int,
        default=10,
        choices=[10, 15, 20],
        help="width of mass bins",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="24Sep25_v12v2_private_signal",
        help="tag for input ntuples",
    )
    parser.add_argument(
        "--years",
        type=str,
        nargs="+",
        default=hh_vars.years,
        choices=hh_vars.years,
        help="years to postprocess",
    )
    parser.add_argument(
        "--training-years",
        nargs="+",
        choices=hh_vars.years,
        help="years used in training",
    )
    parser.add_argument(
        "--mass",
        type=str,
        default="H2PNetMass",
        choices=["H2Msd", "H2PNetMass"],
        help="mass variable to make template",
    )
    parser.add_argument(
        "--bdt-model",
        type=str,
        default="25Feb5_v13_glopartv2_rawmass",
        help="BDT model to load",
    )
    parser.add_argument(
        "--bdt-config",
        type=str,
        default="v13_glopartv2",
        help="BDT model to load",
    )
    parser.add_argument(
        "--txbb",
        type=str,
        default="glopart-v2",
        choices=["pnet-legacy", "pnet-v12", "glopart-v2"],
        help="version of TXbb tagger/mass regression to use",
    )
    parser.add_argument(
        "--txbb-wps",
        type=float,
        nargs=2,
        default=[0.975, 0.82],
        help="TXbb Bin 1, Bin 2 WPs",
    )

    parser.add_argument(
        "--bdt-wps",
        type=float,
        nargs=3,
        default=[0.98, 0.88, 0.03],
        help="BDT Bin 1, Bin 2, Fail WPs",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="abcd",
        choices=["abcd", "sideband"],
        help="method for scanning",
    )

    parser.add_argument("--vbf-txbb-wp", type=float, default=0.95, help="TXbb VBF WP")
    parser.add_argument("--vbf-bdt-wp", type=float, default=0.98, help="BDT VBF WP")

    parser.add_argument(
        "--weight-ttbar-bdt", type=float, default=1.0, help="Weight TTbar discriminator on VBF BDT"
    )

    parser.add_argument(
        "--sig-keys",
        type=str,
        nargs="+",
        default=hh_vars.sig_keys,
        choices=hh_vars.sig_keys,
        help="sig keys for which to make templates",
    )
    parser.add_argument("--pt-first", type=float, default=300, help="pt threshold for leading jet")
    parser.add_argument(
        "--pt-second", type=float, default=250, help="pt threshold for subleading jet"
    )
    parser.add_argument(
        "--fom-vbf-samples",
        type=str,
        nargs="+",
        default=["vbfhh4b-k2v0"],
        help="Samples to use for FOM scan for VBF category",
        choices=["hh4b", "vbfhh4b", "vbfhh4b-k2v0"],
    )
    parser.add_argument(
        "--fom-ggf-samples",
        type=str,
        nargs="+",
        default=["hh4b"],
        help="Samples to use for FOM scan for ggF categories",
        choices=["hh4b", "vbfhh4b"],
    )
    run_utils.add_bool_arg(
        parser,
        "bdt-disc",
        default=True,
        help="use BDT discriminant defined as BDT_ggF/VBF = P_ggF/VBF / (P_ggF/VBF + P_bkg), otherwise use P_ggF/VBF",
    )
    run_utils.add_bool_arg(
        parser,
        "event-list",
        default=False,
        help="generates event list that passes boosted selection for the purpose of overlap removal",
    )
    parser.add_argument(
        "--event-list-dir",
        type=str,
        default="event_lists",
        help="folder to save the event list for each year",
    )
    run_utils.add_bool_arg(parser, "bdt-roc", default=False, help="make BDT ROC curve")
    run_utils.add_bool_arg(parser, "control-plots", default=False, help="make control plots")
    run_utils.add_bool_arg(parser, "fom-scan", default=False, help="run figure of merit scans")
    run_utils.add_bool_arg(parser, "fom-scan-bin1", default=True, help="FOM scan for bin 1")
    run_utils.add_bool_arg(parser, "fom-scan-bin2", default=True, help="FOM scan for bin 2")
    run_utils.add_bool_arg(parser, "fom-scan-vbf", default=False, help="FOM scan for VBF bin")
    run_utils.add_bool_arg(parser, "templates", default=True, help="make templates")
    run_utils.add_bool_arg(parser, "vbf", default=True, help="Add VBF region")
    run_utils.add_bool_arg(
        parser, "vbf-priority", default=False, help="Prioritize the VBF region over ggF Cat 1"
    )
    run_utils.add_bool_arg(
        parser, "correct-vbf-bdt-shape", default=True, help="Correct ttbar BDT_VBF shape"
    )
    run_utils.add_bool_arg(parser, "blind", default=True, help="Blind the analysis")
    run_utils.add_bool_arg(parser, "rerun-inference", default=False, help="Rerun BDT inference")
    run_utils.add_bool_arg(
        parser, "scale-smear", default=False, help="Rerun scaling and smearing of mass variables"
    )
    run_utils.add_bool_arg(
        parser, "dummy-txbb-sfs", default=False, help="use dummy TXbb SFs = 1+/-0.15"
    )

    args = parser.parse_args()

    print(args)
    postprocess_run3(args)

"""
This script loads a pre-trained BDT model, loads data with specified columns and cuts,
and performs inference to obtain BDT scores.

Functions included:
1. `load_events`: Loads the data columns, applies preselection cuts.
2. `apply_cuts`: Applies specific preselection cuts consistent with training.
3. `load_and_score_bdt`: Loads the BDT model, prepares data, applies cuts, and performs inference.

Usage:
    ```
    python bdt_score_test.py --model-name <model_name> --config-name <config_name> --data-path <path_to_data> --year <year>
    ```

Example: python bdt_score_test.py --model-name 24May31_lr_0p02_md_8_AK4Away --config-name 24May31_lr_0p02_md_8_AK4Away --data-path /home/users/dprimosc/data/24Sep25_v12v2_private_signal --year 2022


Arguments:
    --model-name     Name of the model directory containing the trained BDT model file.
    --config-name    Name of the configuration module for data preparation.
    --data-path      Path to the directory with data files for inference.
    --year           Year for data loading configuration.

Returns:
    Prints BDT scores based on the loaded model and input data.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from HH4b.log_utils import log_config
from HH4b.utils import format_columns, get_var_mapping, load_samples

# Configure logging as in Validate BDT script
log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("BDTScoreTest")

jet_collection = "bbFatJet"
num_jets = 2

# Define preselection and mass criteria as in Validate BDT
txbb_preselection = {
    "bbFatJetPNetTXbb": 0.3,
    "bbFatJetPNetTXbbLegacy": 0.8,
    "bbFatJetParTTXbb": 0.3,
}
msd1_preselection = {
    "bbFatJetPNetTXbb": 40,
    "bbFatJetPNetTXbbLegacy": 40,
    "bbFatJetParTTXbb": 40,
}
msd2_preselection = {
    "bbFatJetPNetTXbb": 30,
    "bbFatJetPNetTXbbLegacy": 30,
    "bbFatJetParTTXbb": 30,
}


# Sample directories for QCD and signal datasets as in Validate BDT
def get_sample_dirs(year):
    return {
        year: {
            "qcd": [
                "QCD_HT-1000to1200",
                "QCD_HT-1200to1500",
                "QCD_HT-1500to2000",
                "QCD_HT-2000",
                "QCD_HT-400to600",
                "QCD_HT-600to800",
                "QCD_HT-800to1000",
            ],
        }
    }


def get_sample_dirs_sig(year):
    return {year: {"hh4b": ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00_TuneCP5_13p6TeV?"]}}


# Columns identical to Validate BDT
columns = [
    ("bdt_score", 1),
    ("bdt_score_vbf", 1),
    ("weight", 1),
    ("event", 1),
    ("MET_pt", 1),
    ("bbFatJetTau3OverTau2", 2),
    ("VBFJetPt", 2),
    ("VBFJetEta", 2),
    ("VBFJetPhi", 2),
    ("VBFJetMass", 2),
    ("AK4JetAwayPt", 2),
    ("AK4JetAwayEta", 2),
    ("AK4JetAwayPhi", 2),
    ("AK4JetAwayMass", 2),
    (f"{jet_collection}Pt", num_jets),
    (f"{jet_collection}Msd", num_jets),
    (f"{jet_collection}Eta", num_jets),
    (f"{jet_collection}Phi", num_jets),
    (f"{jet_collection}PNetPXbbLegacy", num_jets),
    (f"{jet_collection}PNetPQCDbLegacy", num_jets),
    (f"{jet_collection}PNetPQCDbbLegacy", num_jets),
    (f"{jet_collection}PNetPQCD0HFLegacy", num_jets),
    (f"{jet_collection}PNetMassLegacy", num_jets),
    (f"{jet_collection}PNetTXbbLegacy", num_jets),
    (f"{jet_collection}PNetTXbb", num_jets),
    (f"{jet_collection}PNetMass", num_jets),
    (f"{jet_collection}PNetQCD0HF", num_jets),
    (f"{jet_collection}PNetQCD1HF", num_jets),
    (f"{jet_collection}PNetQCD2HF", num_jets),
    (f"{jet_collection}ParTmassVis", num_jets),
    (f"{jet_collection}ParTTXbb", num_jets),
    (f"{jet_collection}ParTPXbb", num_jets),
    (f"{jet_collection}ParTPQCD0HF", num_jets),
    (f"{jet_collection}ParTPQCD1HF", num_jets),
    (f"{jet_collection}ParTPQCD2HF", num_jets),
]


def load_events_from_directory(directory_path):
    events_dict = {}
    events_list = []  # List to store events DataFrames

    # Iterate over all parquet files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".parquet"):
            file_path = Path(directory_path) / filename
            # Read the parquet file and append to the list
            events = pd.read_parquet(file_path)
            events_list.append(events)

    # Concatenate all DataFrames in the list
    if events_list:
        combined_events = pd.concat(events_list, ignore_index=True)
        events_dict["hh4b"] = combined_events
        # for key in events.keys():
        #    print(key)
    else:
        print("No parquet files found in the directory.")

    return events_dict


def load_events(path_to_dir, year, jet_coll_pnet):
    """Load events with exact structure and filter settings as in the validate script."""
    txbb_str = f"{jet_collection}{jet_coll_pnet}"
    # mass_str = f"{jet_collection}{jet_coll_mass}"

    # Filters match Validate BDT script
    filters = [
        [
            (f"('{jet_collection}Pt', '0')", ">=", 250),
            (f"('{jet_collection}Pt', '1')", ">=", 250),
        ]
    ]

    sample_dirs = get_sample_dirs(year)
    sample_dirs_sig = get_sample_dirs_sig(year)

    events_dict = {
        **load_samples(
            path_to_dir,
            sample_dirs_sig[year],
            year,
            filters=filters,
            columns=format_columns(columns),
            reorder_txbb=True,
            txbb_str=txbb_str,
            variations=False,
        ),
        **load_samples(
            path_to_dir,
            sample_dirs[year],
            year,
            filters=filters,
            columns=format_columns(columns),
            reorder_txbb=True,
            txbb_str=txbb_str,
            variations=False,
        ),
    }
    return events_dict


def apply_cuts(events_dict, txbb_str, mass_str):
    """Apply cuts identical to the Validate BDT script."""
    for key in events_dict:
        msd1 = events_dict[key]["bbFatJetMsd"][0]
        msd2 = events_dict[key]["bbFatJetMsd"][1]
        pt1 = events_dict[key]["bbFatJetPt"][0]
        pt2 = events_dict[key]["bbFatJetPt"][1]
        txbb1 = events_dict[key][txbb_str][0]
        mass1 = events_dict[key][mass_str][0]
        mass2 = events_dict[key][mass_str][1]
        # h1mass = events_dict[key][mass_str][0]

        events_dict[key] = events_dict[key][
            (pt1 > 250)
            & (pt2 > 250)
            & (txbb1 > txbb_preselection[txbb_str])
            & (msd1 > msd1_preselection[txbb_str])
            & (msd2 > msd2_preselection[txbb_str])
            & (mass1 > 50)
            & (mass2 > 50)
            # & (h1mass > 120)
            # & (h1mass < 130)
        ].copy()
    return events_dict


def get_bdt_score(events_dict, model_name, config_name, jlabel=""):
    """Load BDT model, apply it to events, and get scores."""
    model = xgb.XGBClassifier()
    model.load_model(f"../boosted/bdt_trainings_run3/{model_name}/trained_bdt.model")
    make_bdt_dataframe = importlib.import_module(
        f".{config_name}", package="HH4b.boosted.bdt_trainings_run3"
    )
    scores = {}
    for key in events_dict:
        bdt_events = make_bdt_dataframe.bdt_dataframe(events_dict[key], get_var_mapping(jlabel))
        preds = model.predict_proba(bdt_events)
        if preds.shape[1] == 2:
            scores[key] = preds[:, 1]
        elif preds.shape[1] >= 3:
            scores[key] = preds[:, 0]
    return scores


def main(args):

    # events_dict = load_events(args.data_path, year, "ParTTXbb", "ParTmassVis")
    events_dict = load_events_from_directory(args.data_path)
    events_dict = apply_cuts(
        events_dict, f"{jet_collection}ParTTXbb", f"{jet_collection}ParTmassVis"
    )
    scores = get_bdt_score(events_dict, args.model_name, args.config_name)
    ntuple_scores = events_dict["hh4b"]["bdt_score"]

    ntuple_scores = ntuple_scores.to_numpy().ravel()

    score_diff = np.abs(scores["hh4b"] - ntuple_scores)

    # Create the plot directory if it doesn't exist
    Path("bdt_score_ntuple_comparison").mkdir(parents=True, exist_ok=True)

    # Line plot for direct comparison
    plt.figure(figsize=(10, 6))
    plt.plot(scores["hh4b"], label="BDT Score", color="blue")
    plt.plot(ntuple_scores, label="Ntuple Score", color="green", linestyle="--")

    plt.title("Line Plot Comparison of BDT and Ntuple Scores")
    plt.xlabel("Event Index")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("bdt_score_ntuple_comparison/line_plot_score_comparison.png")
    plt.close()

    # Plot histograms for both score sets
    plt.figure(figsize=(12, 6))

    plt.hist(scores["hh4b"], bins=50, alpha=0.5, label="BDT Score", color="blue")
    plt.hist(ntuple_scores, bins=50, alpha=0.5, label="Ntuple Score", color="green")

    plt.title("Score Distributions")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("bdt_score_ntuple_comparison/histogram_score_distributions.png")
    plt.close()

    # Scatter plot of BDT scores vs. Ntuple scores
    plt.figure(figsize=(8, 8))
    plt.scatter(scores["hh4b"], ntuple_scores, alpha=0.6)
    plt.plot([0, 1], [0, 1], "r--", label="y=x (Perfect Match)")

    plt.title("BDT Scores vs. Ntuple Scores")
    plt.xlabel("BDT Score")
    plt.ylabel("Ntuple Score")
    plt.legend()
    plt.savefig("bdt_score_ntuple_comparison/scatter_plot_scores.png")
    plt.close()

    # Plot histogram of absolute differences between BDT and Ntuple scores
    plt.figure(figsize=(10, 5))
    plt.hist(score_diff, bins=50, color="purple", alpha=0.7)

    plt.title("Absolute Differences Between BDT and Ntuple Scores")
    plt.xlabel("Absolute Difference")
    plt.ylabel("Frequency")
    plt.savefig("bdt_score_ntuple_comparison/histogram_score_differences.png")
    plt.close()

    # Boxplot comparison of BDT and Ntuple scores
    plt.figure(figsize=(8, 6))
    plt.boxplot([scores["hh4b"], ntuple_scores], labels=["BDT Score", "Ntuple Score"])

    plt.title("Boxplot Comparison of Scores")
    plt.ylabel("Score")
    plt.savefig("bdt_score_ntuple_comparison/boxplot_score_comparison.png")
    plt.close()

    # for key, score in scores.items():
    #    print(f"{year} {key} BDT scores: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Model name for BDT model file")
    parser.add_argument("--config-name", required=True, help="Config name for variable mapping")
    parser.add_argument("--data-path", required=True, help="Path to the data directory")
    parser.add_argument("--year", nargs="+", type=str, required=True, help="Year(s) of the dataset")
    args = parser.parse_args()
    main(args)

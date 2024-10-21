from __future__ import annotations

import time

import numpy as np
import pandas as pd
import vector
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence


def bdt_dataframe(events, key_map=lambda x: x):
    """
    Make dataframe with BDT inputs
    NOTE: this function should be saved along with the model for inference usage
    """

    h1 = vector.array(
        {
            "pt": events[key_map("bbFatJetPt")].to_numpy()[:, 0],
            "phi": events[key_map("bbFatJetPhi")].to_numpy()[:, 0],
            "eta": events[key_map("bbFatJetEta")].to_numpy()[:, 0],
            "M": events[key_map("bbFatJetParTmassVis")].to_numpy()[:, 0],
        }
    )
    h2 = vector.array(
        {
            "pt": events[key_map("bbFatJetPt")].to_numpy()[:, 1],
            "phi": events[key_map("bbFatJetPhi")].to_numpy()[:, 1],
            "eta": events[key_map("bbFatJetEta")].to_numpy()[:, 1],
            "M": events[key_map("bbFatJetParTmassVis")].to_numpy()[:, 1],
        }
    )
    hh = h1 + h2

    vbf1 = vector.array(
        {
            "pt": events[(key_map("VBFJetPt"), 0)],
            "phi": events[(key_map("VBFJetPhi"), 0)],
            "eta": events[(key_map("VBFJetEta"), 0)],
            "M": events[(key_map("VBFJetMass"), 0)],
        }
    )

    vbf2 = vector.array(
        {
            "pt": events[(key_map("VBFJetPt"), 1)],
            "phi": events[(key_map("VBFJetPhi"), 1)],
            "eta": events[(key_map("VBFJetEta"), 1)],
            "M": events[(key_map("VBFJetMass"), 1)],
        }
    )

    jj = vbf1 + vbf2

    # AK4JetAway
    ak4away1 = vector.array(
        {
            "pt": events[(key_map("AK4JetAwayPt"), 0)],
            "phi": events[(key_map("AK4JetAwayPhi"), 0)],
            "eta": events[(key_map("AK4JetAwayEta"), 0)],
            "M": events[(key_map("AK4JetAwayMass"), 0)],
        }
    )

    ak4away2 = vector.array(
        {
            "pt": events[(key_map("AK4JetAwayPt"), 1)],
            "phi": events[(key_map("AK4JetAwayPhi"), 1)],
            "eta": events[(key_map("AK4JetAwayEta"), 1)],
            "M": events[(key_map("AK4JetAwayMass"), 1)],
        }
    )

    h1ak4away1 = h1 + ak4away1
    h2ak4away2 = h2 + ak4away2

    df_events = pd.DataFrame(
        {
            # dihiggs system
            key_map("HHPt"): hh.pt,
            key_map("HHeta"): hh.eta,
            key_map("HHmass"): hh.mass,
            # met in the event
            key_map("MET"): events.MET_pt[0],
            # fatjet tau32
            key_map("H1T32"): events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 0],
            key_map("H2T32"): events[key_map("bbFatJetTau3OverTau2")].to_numpy()[:, 1],
            # fatjet mass
            key_map("H1Mass"): events[key_map("bbFatJetParTmassVis")].to_numpy()[:, 0],
            # fatjet kinematics
            key_map("H1Pt"): h1.pt,
            key_map("H2Pt"): h2.pt,
            key_map("H1eta"): h1.eta,
            # xbb
            key_map("H1Xbb"): events[key_map("bbFatJetParTPXbb")].to_numpy()[:, 0],
            key_map("H1QCDb"): events[key_map("bbFatJetParTPQCD1HF")].to_numpy()[:, 0],
            key_map("H1QCDbb"): events[key_map("bbFatJetParTPQCD2HF")].to_numpy()[:, 0],
            key_map("H1QCDothers"): events[key_map("bbFatJetParTPQCD0HF")].to_numpy()[:, 0],
            # ratios
            key_map("H1Pt_HHmass"): h1.pt / hh.mass,
            key_map("H2Pt_HHmass"): h2.pt / hh.mass,
            key_map("H1Pt/H2Pt"): h1.pt / h2.pt,
            # vbf mjj and eta_jj
            key_map("VBFjjMass"): jj.mass,
            key_map("VBFjjDeltaEta"): np.abs(vbf1.eta - vbf2.eta),
            # AK4JetAway
            key_map("H1AK4JetAway1dR"): h1.deltaR(ak4away1),
            key_map("H2AK4JetAway2dR"): h2.deltaR(ak4away2),
            key_map("H1AK4JetAway1mass"): h1ak4away1.mass,
            key_map("H2AK4JetAway2mass"): h2ak4away2.mass,
        }
    )

    # Add the TDA feature to the dataframe
    df_events = add_tda_feature(df_events)
    print("TDA feature added to the dataframe.")

    return df_events


def compute_tda_feature(data: pd.DataFrame) -> np.ndarray:
    """
    Compute persistent homology and return a TDA-based feature (e.g., persistence entropy).
    Args:
        data: A DataFrame containing selected features for TDA.
    Returns:
        A NumPy array containing the computed TDA feature (flattened).
    """
    print("Computing TDA feature...")
    start_time = time.time()

    # Convert selected features into a point cloud for TDA
    point_cloud = data[["VBFjjDeltaEta", "H1AK4JetAway1dR"]].to_numpy()

    # Initialize the Vietoris-Rips persistence pipeline
    vr = VietorisRipsPersistence(homology_dimensions=[0, 1])

    # Compute the persistence diagrams
    diagrams = vr.fit_transform(point_cloud[None, :, :])  # Add batch dimension

    # Extract summary statistics (e.g., persistence entropy)
    pe = PersistenceEntropy()
    tda_feature = pe.fit_transform(diagrams)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Log the computation time into a text file
    with open("tda_computation_time.txt", "a") as log_file:
        log_file.write(f"TDA computation time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n")
    return tda_feature.flatten()


def add_tda_feature(events: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the TDA feature to the input events DataFrame.
    """
    tda_feature = compute_tda_feature(events)
    print("Done.")
    events["TDA_Feature"] = tda_feature
    return events

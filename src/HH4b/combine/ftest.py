"""
Ftest script for HH4b analysis, adopted from FTest.ipynb.

TODO: this is a wip

Author: Daniel Primosch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot
from scipy import stats

import HH4b

# Use CMS plotting style
plt.style.use(hep.style.CMS)
hep.style.use("CMS")
formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
plt.rcParams.update({"font.size": 20})


def p_value(data_ts: float, toy_ts: list[float]) -> float:
    """
    Calculate the p-value for data_ts given a distribution of toy_ts
    """
    return np.mean(toy_ts >= data_ts)


def F_statistic(
    ts_low: np.ndarray,
    ts_high: np.ndarray,
    ord_low: int,
    ord_high: int,
    num_bins: int = 10 * 14,
    dim: int = 2,
):
    """
    Calculate the F-statistic used in the F-test.

    Parameters
    ----------
    ts_low: np.ndarray
        Test statistics for the lower-order fit
    ts_high: np.ndarray
        Test statistics for the higher-order fit
    ord_low: int
        Polynomial/order for the low model
    ord_high: int
        Polynomial/order for the high model
    num_bins: int
        Number of bins in the fit
    dim: int
        Dimensionality (default 2 for 2D fit, for instance)

    Returns
    -------
    np.ndarray
        The F-statistic
    """
    numerator = (ts_low - ts_high) / (ord_high - ord_low)
    denominator = ts_high / (num_bins - (ord_high + dim))
    return numerator / denominator


def plot_tests(
    data_ts: float,
    toy_ts: np.ndarray,
    name: str,
    plot_dir: Path,
    category_label: str = "",
    title: str = None,
    bins: int = 15,
    fit: str = None,
    fdof2: int = None,
    xlim: float = None,
):
    """
    Make a histogram plot of the test statistics from toys, overlay the data TS,
    and optionally overlay a fit (chi^2 or F distribution).

    data_ts: float
        Test statistic value for data
    toy_ts: np.ndarray
        Array of test statistic values from toys
    name: str
        Name of the output plot (without extension)
    plot_dir: Path
        Directory to save plots
    category_label: str
        Label for the plot legend (e.g. category name)
    title: str
        Plot title
    bins: int
        Number of bins in the histogram
    fit: str
        Which fit to overlay: 'chi2' or 'f', or None
    fdof2: int
        Degrees of freedom for F distribution (when fit='f')
    xlim: float
        Optional upper bound for plotting
    """
    plot_max = max(np.max(toy_ts), data_ts)
    plot_min = 0  # we only expect TS > 0
    pval = p_value(data_ts, toy_ts)

    plt.figure(figsize=(12, 8))
    h = plt.hist(
        toy_ts,
        np.linspace(plot_min, plot_max if xlim is None else xlim, bins + 1),
        color="#8C8C8C",
        histtype="step",
        label=f"{len(toy_ts)} Toys",
    )
    plt.axvline(data_ts, color="#FF502E", linestyle=":", label=rf"Data ($p$-value = {pval:.2f})")

    # Optionally overlay a fit
    if fit is not None:
        x = np.linspace(plot_min + 0.01, plot_max if xlim is None else xlim, 200)

        if fit == "chi2":
            # Fit to a Chi-square distribution using MLE or method of moments
            # For a simpler approach, you might do e.g. manual guess or use stats.chi2.fit
            # For demonstration, let's do a naive method-of-moments approach
            # or you can do something like:
            shape_guess = np.mean(toy_ts)  # naive guess = mean as dof
            pdf = stats.chi2.pdf(x, df=shape_guess)
            label = rf"$\chi^2(\mathrm{{DoF}}={shape_guess:.2f})$"

            # If you wanted to do a full parametric fit, you can use:
            # shape, loc, scale = stats.chi2.fit(toy_ts, floc=0, fscale=1)
            # pdf = stats.chi2.pdf(x, shape, loc, scale)
            # label = rf"$\chi^2_{{\mathrm{{DoF}} = {shape:.2f}}}$ fit"

        elif fit == "f":
            # For an F distribution, we fix the numerator dof = 1, user must supply denominator dof
            if fdof2 is None:
                raise ValueError("fdof2 must be provided when fit='f'")
            pdf = stats.f.pdf(x, 1, fdof2)
            label = rf"$F(1, {fdof2})$"

        else:
            raise ValueError("Invalid fit type. Choose either 'chi2' or 'f'.")

        plt.plot(
            x,
            pdf * (np.max(h[0]) / np.max(pdf)),
            color="#1f78b4",
            linestyle="--",
            label=label,
        )

    hep.cms.label(
        "Work in Progress",
        data=True,
        lumi=61,
        year=None,
    )

    plt.legend(title=category_label)
    if title:
        plt.title(title)
    plt.ylabel("Number of Toys")
    plt.xlabel("Test Statistic")

    outname = plot_dir / f"{name}.pdf"
    plt.savefig(outname, bbox_inches="tight")
    print(f"Saved plot {outname}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Script that performs GOF (Goodness of Fit) and F-tests, and plots the results."
    )
    parser.add_argument(
        "--passbin",
        default="passvbf",
        type=str,
        help="Category bin name, e.g. passvbf, passbin0, passbin1, etc.",
    )
    parser.add_argument(
        "--test-orders",
        nargs="+",
        default=[0],
        type=int,
        help="List of integer orders to test. Typically we compare order 'o1' vs 'o1+1'.",
    )
    parser.add_argument(
        "--f-tests-dir",
        default="run3-bdt-november11-glopartv2",
        type=str,
        help="Subdirectory for the F-test cards, e.g. 'run3-bdt-november11-glopartv2'.",
    )
    parser.add_argument(
        "--cards-dir-second",
        default="run3-bdt-july6-unblind-ntf0011",
        type=str,
        help="Subdirectory for the second GOF test (combined), if applicable.",
    )
    parser.add_argument(
        "--no-second-test",
        action="store_true",
        default=False,
        help="If set, skip the second test with passbin0. Only run the first test.",
    )

    args = parser.parse_args()

    # Retrieve top-level directory from HH4b (adapt if you don't have HH4b)
    MAIN_DIR = Path(HH4b.__file__).parents[2]

    # 1) ----------------------------------------------------------------------
    # F-test/GOF for a given passbin and "f-tests-dir"
    passbin = args.passbin
    plot_dir = MAIN_DIR / f"plots/FTests/{args.f_tests_dir}/{passbin}"
    plot_dir.mkdir(exist_ok=True, parents=True)

    local_cards_dir = MAIN_DIR / f"src/HH4b/cards/f_tests/{args.f_tests_dir}"

    test_statistics = {}
    test_orders = args.test_orders  # e.g. [0], or [1], or [0,2], etc.

    # We'll just do logic that compares each o1 with (o1+1).
    # If you want to do arbitrary pairs, adapt as needed.
    for o1 in test_orders:
        tlabel = f"{o1}"
        tdict = {"toys": {}, "data": {}, "ftoys": {}, "fdata": {}}

        for nTF in [o1, o1 + 1]:
            tflabel = f"{nTF}"

            # Toys
            # in your original notebook you had e.g.:
            # "higgsCombineToys{tlabel}.GoodnessOfFit.mH125.*.root"
            # We'll replicate that logic here.

            toy_files = str(
                local_cards_dir
                / f"{passbin}_nTF_{nTF}"
                / f"higgsCombineToys{tlabel}.GoodnessOfFit.mH125.*.root"
            )
            file_toys = uproot.concatenate(toy_files)
            tdict["toys"][tflabel] = np.array(file_toys["limit"])

            # Data
            data_file = str(
                local_cards_dir
                / f"{passbin}_nTF_{nTF}"
                / "higgsCombineData.GoodnessOfFit.mH125.root"
            )
            file_data = uproot.concatenate(data_file)
            tdict["data"][tflabel] = file_data["limit"][0]

            if nTF != o1:
                tdict["ftoys"][tflabel] = F_statistic(
                    tdict["toys"][tlabel],
                    tdict["toys"][tflabel],
                    o1,
                    nTF,
                )
                tdict["fdata"][tflabel] = F_statistic(
                    tdict["data"][tlabel],
                    tdict["data"][tflabel],
                    o1,
                    nTF,
                )

        test_statistics[tlabel] = tdict

    # Now make plots
    # Example: for each o1 in test_orders, make the GOF and F-test plots
    for o1 in test_orders:
        tlabel = f"{o1}"

        # GOF (Goodness-of-Fit) for the model with order o1
        data_ts = test_statistics[tlabel]["data"][tlabel]
        toy_ts = test_statistics[tlabel]["toys"][tlabel]
        plot_tests(
            data_ts=data_ts,
            toy_ts=toy_ts,
            name=f"gof_{tlabel}",
            plot_dir=plot_dir,
            category_label=f"{passbin}",
            fit="chi2",  # if you want to overlay a chi^2 shape
            title=f"GOF test, order {o1}",
            bins=20,
        )

        # F-test comparing order o1 and (o1+1)
        ord1 = o1 + 1
        tflabel = f"{ord1}"
        data_ts = test_statistics[tlabel]["fdata"][tflabel]
        toy_ts = test_statistics[tlabel]["ftoys"][tflabel]
        plot_tests(
            data_ts=data_ts,
            toy_ts=toy_ts,
            name=f"f{o1}_{ord1}",
            plot_dir=plot_dir,
            category_label=f"{passbin}",
            title=f"F-test: order {o1} vs. {ord1}",
            fit="f",
            fdof2=(10 * 14 - (ord1 + 2)),  # Example: if dim=2, adapt as needed
            xlim=100,
        )

    # 2) ----------------------------------------------------------------------
    # Optionally do a second GOF test with e.g. "passbin0" and a different cards dir
    if not args.no_second_test:
        passbin0 = "passbin0"
        local_cards_dir_second = MAIN_DIR / f"src/HH4b/cards/{args.cards_dir_second}"
        plot_dir_second = MAIN_DIR / f"plots/{args.cards_dir_second}"
        plot_dir_second.mkdir(exist_ok=True, parents=True)

        tdict = {}
        # test statistics for toys
        toy_files = str(local_cards_dir_second / "higgsCombineToys.GoodnessOfFit.mH125.*.root")
        file_toys = uproot.concatenate(toy_files)
        quantileExpected = np.array(file_toys["quantileExpected"])

        tdict["toys"] = np.array(file_toys["limit"])[quantileExpected > -2]

        # data test statistic
        data_file = str(local_cards_dir_second / "higgsCombineData.GoodnessOfFit.mH125.root")
        file_data = uproot.concatenate(data_file)
        tdict["data"] = file_data["limit"][0]

        data_ts = tdict["data"]
        toy_ts = tdict["toys"]
        plot_tests(
            data_ts=data_ts,
            toy_ts=toy_ts,
            name="gof_combined",
            plot_dir=plot_dir_second,
            category_label=passbin0,
            fit="chi2",
            title="GOF test, combined",
            bins=20,
        )


if __name__ == "__main__":
    main()

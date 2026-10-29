from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path("results")
OUTPUT = RESULTS / "figures"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT / f"{name}.png"
    pdf_path = OUTPUT / f"{name}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def figure_1_baseline_comparison():
    """
    Main detector comparison on clean test data.
    """

    names = [
        "Isolation\nForest",
        "Fixed-step\nKalman",
        "Δt-aware\nKalman",
        "Standalone\nGRU",
        "Kalman +\nGRU",
        "GapGuard",
    ]

    f1 = [
        0.1594,
        0.0650,
        0.0924,
        0.0763,
        0.0678,
        0.1054,
    ]

    pr_auc = [
        0.1051,
        0.1186,
        0.1314,
        0.1087,
        0.1258,
        np.nan,
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.bar(
        x - width / 2,
        f1,
        width,
        label="F1"
    )

    ax.bar(
        x + width / 2,
        pr_auc,
        width,
        label="PR-AUC"
    )

    ax.set_ylabel("Score")
    ax.set_title("Clean-test comparison of anomaly detectors")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 0.18)
    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure1_baseline_comparison"
    )


def figure_2_packet_loss_fpr():
    """
    Frozen clean calibration under packet loss.
    """

    loss_rates = np.array([10, 30, 50])

    random_global = np.array([
        0.0157,
        0.0301,
        0.0534,
    ])

    random_gapguard = np.array([
        0.0124,
        0.0207,
        0.0377,
    ])

    burst_global = np.array([
        0.0144,
        0.0258,
        0.0449,
    ])

    burst_gapguard = np.array([
        0.0133,
        0.0218,
        0.0375,
    ])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(
        loss_rates,
        random_global * 100,
        marker="o",
        linestyle="--",
        label="Random loss — global"
    )

    ax.plot(
        loss_rates,
        random_gapguard * 100,
        marker="o",
        label="Random loss — GapGuard"
    )

    ax.plot(
        loss_rates,
        burst_global * 100,
        marker="s",
        linestyle="--",
        label="Burst loss — global"
    )

    ax.plot(
        loss_rates,
        burst_gapguard * 100,
        marker="s",
        label="Burst loss — GapGuard"
    )

    ax.set_xlabel("Packet-loss rate (%)")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title(
        "False-positive inflation under packet-loss shift"
    )

    ax.set_xticks(loss_rates)
    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure2_packet_loss_fpr"
    )


def figure_3_evidence_fpr():
    """
    Clean-test FPR by evidence stratum.
    """

    evidence = [
        "Low",
        "Medium",
        "High",
    ]

    global_fpr = np.array([
        0.0305652,
        0.0003460,
        0.0001177,
    ])

    conditioned_fpr = np.array([
        0.0105677,
        0.0106263,
        0.0095767,
    ])

    x = np.arange(len(evidence))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.bar(
        x - width / 2,
        global_fpr * 100,
        width,
        label="Global threshold"
    )

    ax.bar(
        x + width / 2,
        conditioned_fpr * 100,
        width,
        label="Evidence-conditioned"
    )

    ax.axhline(
        1.0,
        linestyle=":",
        label="Target FPR = 1%"
    )

    ax.set_xlabel("Evidence quality")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title(
        "Evidence-stratified false-positive rate"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(evidence)

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure3_evidence_stratified_fpr"
    )


def main():

    print("=" * 80)
    print("STAGE 4.3 — FINAL PAPER FIGURES")
    print("=" * 80)

    figure_1_baseline_comparison()
    figure_2_packet_loss_fpr()
    figure_3_evidence_fpr()

    print("\nSTAGE 4.3 COMPLETE")


if __name__ == "__main__":
    main()
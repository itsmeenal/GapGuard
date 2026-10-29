from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_DIR = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "random"
)

BURST_STATS = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "burst"
    / "burst_loss_statistics.csv"
)


# ============================================================
# LOAD RANDOM-LOSS RESULTS
# ============================================================

def load_random_results():

    frames = []

    for loss_percent in [10, 30, 50]:

        stats_file = (
            RANDOM_DIR
            / f"loss_{loss_percent}"
            / "loss_statistics.csv"
        )

        if not stats_file.exists():
            print(
                f"Random statistics file not found:\n"
                f"{stats_file}"
            )
            continue

        df = pd.read_csv(stats_file)

        # The original random-loss simulator does not store
        # target_loss_rate explicitly, so recover it from
        # the folder name.
        df["target_loss_rate"] = loss_percent / 100.0

        frames.append(df)

    if not frames:
        return None

    return pd.concat(
        frames,
        ignore_index=True
    )


# ============================================================
# LOAD BURST-LOSS RESULTS
# ============================================================

def load_burst_results():

    if not BURST_STATS.exists():

        print(
            "Burst-loss statistics file not found:"
        )

        print(BURST_STATS)

        return None

    return pd.read_csv(BURST_STATS)


# ============================================================
# STANDARDIZE RANDOM STATISTICS
# ============================================================

def standardize_random_columns(df):

    if df is None:
        return None

    print()
    print("Random-loss columns found:")
    print(list(df.columns))

    # --------------------------------------------------------
    # Actual loss
    # --------------------------------------------------------

    if "actual_loss_rate" not in df.columns:

        if "actual_loss" in df.columns:

            df["actual_loss_rate"] = df["actual_loss"]

        elif "loss_rate" in df.columns:

            df["actual_loss_rate"] = df["loss_rate"]

        else:

            # Reconstruct from original and received rows.
            df["actual_loss_rate"] = (
                (
                    df["original_rows"]
                    - df["received_rows"]
                )
                / df["original_rows"]
            )

    # --------------------------------------------------------
    # Lost attack rows
    # --------------------------------------------------------

    if "lost_attack_rows" not in df.columns:

        df["lost_attack_rows"] = (
            df["original_attack_rows"]
            - df["received_attack_rows"]
        )

    # --------------------------------------------------------
    # Make sure observability exists
    # --------------------------------------------------------

    if "attack_observability" not in df.columns:

        df["attack_observability"] = np.where(
            df["original_attack_rows"] > 0,

            df["received_attack_rows"]
            / df["original_attack_rows"],

            np.nan
        )

    return df


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_dataset(df, name):

    print()
    print("=" * 60)
    print(f"{name.upper()} LOSS VALIDATION")
    print("=" * 60)

    if df is None:

        print("No data available.")

        return

    print(
        f"Rows in statistics: {len(df)}"
    )

    required_columns = [
        "target_loss_rate",
        "original_rows",
        "received_rows",
        "lost_rows",
        "actual_loss_rate",
        "original_attack_rows",
        "received_attack_rows",
        "lost_attack_rows",
        "attack_observability",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print(
            "ERROR — Missing columns:"
        )

        print(missing)

        return

    print(
        "PASS — all required columns are present."
    )

    # --------------------------------------------------------
    # Row accounting
    # --------------------------------------------------------

    row_accounting_error = (
        df["original_rows"]
        - df["received_rows"]
        - df["lost_rows"]
    )

    if np.allclose(
        row_accounting_error,
        0
    ):

        print(
            "PASS — row accounting is correct."
        )

    else:

        print(
            "FAIL — row accounting errors found:",
            int(
                (row_accounting_error != 0).sum()
            )
        )

    # --------------------------------------------------------
    # Attack accounting
    # --------------------------------------------------------

    attack_accounting_error = (
        df["original_attack_rows"]
        - df["received_attack_rows"]
        - df["lost_attack_rows"]
    )

    if np.allclose(
        attack_accounting_error,
        0
    ):

        print(
            "PASS — attack-row accounting is correct."
        )

    else:

        print(
            "FAIL — attack accounting errors found:",
            int(
                (attack_accounting_error != 0).sum()
            )
        )

    # --------------------------------------------------------
    # Observability formula
    # --------------------------------------------------------

    expected_observability = np.where(
        df["original_attack_rows"] > 0,

        df["received_attack_rows"]
        / df["original_attack_rows"],

        np.nan
    )

    valid = (
        ~np.isnan(expected_observability)
        &
        ~np.isnan(
            df["attack_observability"]
        )
    )

    if valid.any():

        max_error = np.max(
            np.abs(
                expected_observability[valid]
                -
                df.loc[
                    valid,
                    "attack_observability"
                ]
            )
        )

        if max_error < 1e-10:

            print(
                "PASS — attack observability "
                "formula is correct."
            )

        else:

            print(
                "FAIL — attack observability "
                "mismatch."
            )

            print(
                f"Maximum error: {max_error}"
            )


# ============================================================
# LOSS-LEVEL SUMMARY
# ============================================================

def print_summary(df, name):

    if df is None:
        return None

    print()
    print("-" * 60)
    print(
        f"{name} LOSS — SUMMARY BY LOSS LEVEL"
    )
    print("-" * 60)

    summary = (
        df
        .groupby("target_loss_rate")
        .agg(
            files=(
                "target_loss_rate",
                "count"
            ),

            mean_actual_loss=(
                "actual_loss_rate",
                "mean"
            ),

            mean_original_rows=(
                "original_rows",
                "mean"
            ),

            mean_received_rows=(
                "received_rows",
                "mean"
            ),

            mean_original_attack_rows=(
                "original_attack_rows",
                "mean"
            ),

            mean_received_attack_rows=(
                "received_attack_rows",
                "mean"
            ),

            mean_lost_attack_rows=(
                "lost_attack_rows",
                "mean"
            ),

            mean_attack_observability=(
                "attack_observability",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# OBSERVABILITY TREND
# ============================================================

def check_observability_trend(
    df,
    name
):

    if df is None:
        return

    summary = (
        df
        .groupby("target_loss_rate")
        ["attack_observability"]
        .mean()
        .sort_index()
    )

    values = summary.values

    print()
    print("-" * 60)
    print(
        f"{name} LOSS — OBSERVABILITY TREND"
    )
    print("-" * 60)

    for loss_rate, observability in summary.items():

        print(
            f"{loss_rate * 100:.0f}% loss "
            f"-> "
            f"{observability:.4f} "
            f"({observability * 100:.2f}%)"
        )

    is_decreasing = all(
        values[i] >= values[i + 1]
        for i in range(
            len(values) - 1
        )
    )

    if is_decreasing:

        print()
        print(
            "PASS — attack observability "
            "decreases as packet loss increases."
        )

    else:

        print()
        print(
            "WARNING — observability is not "
            "monotonically decreasing."
        )


# ============================================================
# RANDOM VS BURST COMPARISON
# ============================================================

def compare_random_and_burst(
    random_df,
    burst_df
):

    if (
        random_df is None
        or burst_df is None
    ):
        return

    print()
    print("=" * 60)
    print(
        "RANDOM VS BURST LOSS COMPARISON"
    )
    print("=" * 60)

    random_summary = (
        random_df
        .groupby("target_loss_rate")
        ["attack_observability"]
        .mean()
        .rename("random")
    )

    burst_summary = (
        burst_df
        .groupby("target_loss_rate")
        ["attack_observability"]
        .mean()
        .rename("burst")
    )

    comparison = pd.concat(
        [
            random_summary,
            burst_summary
        ],
        axis=1
    )

    comparison["difference"] = (
        comparison["burst"]
        -
        comparison["random"]
    )

    print(
        comparison.to_string()
    )

    print()
    print(
        "Difference = burst observability "
        "minus random observability."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(
        "STEP 2.6 — ATTACK OBSERVABILITY VALIDATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    random_df = load_random_results()

    burst_df = load_burst_results()

    # --------------------------------------------------------
    # Standardize random-loss schema
    # --------------------------------------------------------

    random_df = standardize_random_columns(
        random_df
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        random_df,
        "Random"
    )

    validate_dataset(
        burst_df,
        "Burst"
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print_summary(
        random_df,
        "Random"
    )

    print_summary(
        burst_df,
        "Burst"
    )

    # --------------------------------------------------------
    # Trends
    # --------------------------------------------------------

    check_observability_trend(
        random_df,
        "Random"
    )

    check_observability_trend(
        burst_df,
        "Burst"
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    compare_random_and_burst(
        random_df,
        burst_df
    )

    print()
    print("=" * 60)
    print(
        "STEP 2.6 COMPLETE"
    )
    print("=" * 60)
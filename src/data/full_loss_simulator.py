from pathlib import Path
import argparse
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("data/processed/trajectories")
DEFAULT_OUTPUT_ROOT = Path("data/simulated/full")

LOSS_RATES = [0.10, 0.30, 0.50]
BASE_SEED = 42


# ============================================================
# REPRODUCIBLE SEED
# ============================================================

def make_seed(file_name, loss_type, loss_rate):
    """
    Creates a deterministic seed for each:
        file + loss type + loss rate

    We use SHA256 instead of Python's built-in hash()
    because Python hash values can change between sessions.
    """

    key = f"{BASE_SEED}|{file_name}|{loss_type}|{loss_rate}"

    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:4], byteorder="little")


# ============================================================
# RANDOM LOSS
# ============================================================

def apply_random_loss(df, loss_rate, seed):
    """
    Randomly removes approximately loss_rate fraction of rows.

    Important:
    - First observation is always retained.
    - Rows are removed without replacement.
    """

    n = len(df)

    if n <= 1:
        return df.copy(), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)

    removable_indices = np.arange(1, n)

    target_lost = int(round((n - 1) * loss_rate))

    target_lost = min(target_lost, len(removable_indices))

    if target_lost == 0:
        lost_mask = np.zeros(n, dtype=bool)
        return df.copy(), lost_mask

    lost_indices = rng.choice(
        removable_indices,
        size=target_lost,
        replace=False
    )

    lost_mask = np.zeros(n, dtype=bool)
    lost_mask[lost_indices] = True

    received_df = df.loc[~lost_mask].copy()

    return received_df, lost_mask


# ============================================================
# BURST LOSS
# ============================================================

def apply_burst_loss(
    df,
    loss_rate,
    seed,
    min_burst=2,
    max_burst=8
):
    """
    Removes observations in bursts.

    This follows the same general design as our Step 2.5 test:
    - first observation is always retained
    - target overall loss is approximately exact
    - individual generated bursts are 2-8 rows
    - bursts may overlap/touch, so final contiguous missing
      runs can occasionally be longer than 8
    """

    n = len(df)

    if n <= 1:
        return df.copy(), np.zeros(n, dtype=bool)

    rng = np.random.default_rng(seed)

    target_lost = int(round((n - 1) * loss_rate))
    target_lost = min(target_lost, n - 1)

    lost_mask = np.zeros(n, dtype=bool)

    attempts = 0
    max_attempts = max(1000, n * 100)

    while lost_mask.sum() < target_lost and attempts < max_attempts:

        attempts += 1

        remaining_needed = target_lost - int(lost_mask.sum())

        if remaining_needed <= 0:
            break

        burst_length = int(
            rng.integers(
                min_burst,
                max_burst + 1
            )
        )

        burst_length = min(
            burst_length,
            remaining_needed,
            n - 1
        )

        if burst_length <= 0:
            break

        max_start = n - burst_length

        if max_start < 1:
            break

        start = int(
            rng.integers(
                1,
                max_start + 1
            )
        )

        end = start + burst_length

        candidate_indices = np.arange(start, end)

        # Only mark observations that are not already lost.
        new_indices = candidate_indices[
            ~lost_mask[candidate_indices]
        ]

        if len(new_indices) > remaining_needed:
            new_indices = new_indices[:remaining_needed]

        lost_mask[new_indices] = True

    # Safety fallback:
    # If overlap prevented us from reaching exact target,
    # randomly remove remaining available observations.
    current_lost = int(lost_mask.sum())

    if current_lost < target_lost:

        needed = target_lost - current_lost

        available = np.where(~lost_mask)[0]
        available = available[available != 0]

        if len(available) > 0:

            needed = min(needed, len(available))

            extra = rng.choice(
                available,
                size=needed,
                replace=False
            )

            lost_mask[extra] = True

    received_df = df.loc[~lost_mask].copy()

    return received_df, lost_mask


# ============================================================
# BURST STATISTICS
# ============================================================

def get_missing_runs(lost_mask):
    """
    Returns lengths of final contiguous missing runs.
    """

    runs = []
    current_run = 0

    for lost in lost_mask:

        if lost:
            current_run += 1

        else:
            if current_run > 0:
                runs.append(current_run)
                current_run = 0

    if current_run > 0:
        runs.append(current_run)

    return runs


# ============================================================
# STATISTICS FOR ONE TRAJECTORY
# ============================================================

def calculate_statistics(
    original_df,
    received_df,
    lost_mask,
    file_name,
    loss_type,
    target_loss_rate
):
    original_rows = len(original_df)
    received_rows = len(received_df)
    lost_rows = original_rows - received_rows

    if original_rows > 0:
        actual_loss_fraction = lost_rows / original_rows
    else:
        actual_loss_fraction = 0.0

    # Attack rows before loss
    original_attack_rows = int(
        (original_df["label"] != 0).sum()
    )

    # Attack rows that remain observable
    received_attack_rows = int(
        (received_df["label"] != 0).sum()
    )

    lost_attack_rows = (
        original_attack_rows - received_attack_rows
    )

    if original_attack_rows > 0:
        attack_observability = (
            received_attack_rows / original_attack_rows
        )
    else:
        attack_observability = np.nan

    stats = {
        "file": file_name,
        "loss_type": loss_type,
        "target_loss_rate": target_loss_rate,
        "original_rows": original_rows,
        "received_rows": received_rows,
        "lost_rows": lost_rows,
        "actual_loss_fraction": actual_loss_fraction,
        "original_attack_rows": original_attack_rows,
        "received_attack_rows": received_attack_rows,
        "lost_attack_rows": lost_attack_rows,
        "attack_observability": attack_observability,
    }

    if loss_type == "burst":

        runs = get_missing_runs(lost_mask)

        stats["number_of_missing_runs"] = len(runs)

        if runs:
            stats["mean_missing_run_length"] = float(
                np.mean(runs)
            )
            stats["max_missing_run_length"] = int(
                np.max(runs)
            )
        else:
            stats["mean_missing_run_length"] = 0.0
            stats["max_missing_run_length"] = 0

    return stats


# ============================================================
# PROCESS ONE CONDITION
# ============================================================

def process_condition(
    files,
    loss_type,
    loss_rate,
    output_root
):
    rate_percent = int(loss_rate * 100)

    output_dir = (
        output_root
        / loss_type
        / f"loss_{rate_percent}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics = []

    total_files = len(files)

    print()
    print("=" * 65)
    print(
        f"{loss_type.upper()} LOSS — "
        f"{rate_percent}%"
    )
    print("=" * 65)

    for i, input_path in enumerate(files, start=1):

        df = pd.read_csv(input_path)

        seed = make_seed(
            input_path.name,
            loss_type,
            loss_rate
        )

        if loss_type == "random":

            received_df, lost_mask = apply_random_loss(
                df,
                loss_rate,
                seed
            )

        elif loss_type == "burst":

            received_df, lost_mask = apply_burst_loss(
                df,
                loss_rate,
                seed
            )

        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )

        output_path = output_dir / input_path.name

        received_df.to_csv(
            output_path,
            index=False
        )

        stats = calculate_statistics(
            original_df=df,
            received_df=received_df,
            lost_mask=lost_mask,
            file_name=input_path.name,
            loss_type=loss_type,
            target_loss_rate=loss_rate
        )

        statistics.append(stats)

        if (
            i == 1
            or i % 100 == 0
            or i == total_files
        ):
            print(
                f"Processed {i:,} / "
                f"{total_files:,}"
            )

    stats_df = pd.DataFrame(statistics)

    stats_path = (
        output_dir
        / "loss_statistics.csv"
    )

    stats_df.to_csv(
        stats_path,
        index=False
    )

    print(
        f"Statistics saved: {stats_path}"
    )

    return stats_df


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Full GapGuard packet-loss simulator"
        )
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help=(
            "Maximum number of trajectories to process. "
            "0 means all trajectories."
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory."
    )

    args = parser.parse_args()

    output_root = Path(args.output_root)

    files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No trajectory CSV files found in {INPUT_DIR}"
        )

    if args.max_files > 0:
        files = files[:args.max_files]

    print()
    print("=" * 65)
    print("GAPGUARD — FULL LOSS SIMULATOR")
    print("=" * 65)

    print(
        f"Input trajectories : {len(files):,}"
    )

    print(
        f"Output root        : {output_root}"
    )

    print(
        f"Base seed          : {BASE_SEED}"
    )

    all_statistics = []

    for loss_type in ["random", "burst"]:

        for loss_rate in LOSS_RATES:

            stats_df = process_condition(
                files=files,
                loss_type=loss_type,
                loss_rate=loss_rate,
                output_root=output_root
            )

            all_statistics.append(stats_df)

    combined = pd.concat(
        all_statistics,
        ignore_index=True
    )

    combined_path = (
        output_root
        / "combined_loss_statistics.csv"
    )

    combined.to_csv(
        combined_path,
        index=False
    )

    print()
    print("=" * 65)
    print("SIMULATION COMPLETE")
    print("=" * 65)

    print(
        f"Total condition-file pairs: "
        f"{len(combined):,}"
    )

    print(
        f"Combined statistics: "
        f"{combined_path}"
    )

    print()

    summary = (
        combined
        .groupby(
            ["loss_type", "target_loss_rate"]
        )
        .agg(
            files=("file", "count"),
            mean_actual_loss=(
                "actual_loss_fraction",
                "mean"
            ),
            mean_rows_before=(
                "original_rows",
                "mean"
            ),
            mean_rows_after=(
                "received_rows",
                "mean"
            ),
            mean_attack_observability=(
                "attack_observability",
                "mean"
            )
        )
        .reset_index()
    )

    print("SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
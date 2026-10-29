from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "trajectories"

OUTPUT_BASE_DIR = PROJECT_ROOT / "data" / "simulated" / "burst"

LOSS_RATES = [0.10, 0.30, 0.50]

# During testing, process only 100 trajectories.
# Later we will increase this after validation.
MAX_TEST_FILES = 100

# Burst length will normally be between 2 and 8 rows.
MIN_BURST_LENGTH = 2
MAX_BURST_LENGTH = 8

# Fixed seed makes the experiment reproducible.
RANDOM_SEED = 42


# ============================================================
# HELPER: CREATE BURST-LOSS MASK
# ============================================================

def create_burst_loss_mask(n_rows, target_loss_rate, rng):
    """
    Create a boolean mask indicating which observations should be removed.

    True  = remove this observation
    False = keep this observation

    Rules:
    - First observation is always kept.
    - Loss occurs in consecutive bursts.
    - Burst length is normally 2-8 rows.
    - The function safely handles short trajectories.
    """

    if n_rows <= 1:
        return np.zeros(n_rows, dtype=bool)

    # Number of observations we want to remove approximately.
    target_lost = int(round((n_rows - 1) * target_loss_rate))

    if target_lost <= 0:
        return np.zeros(n_rows, dtype=bool)

    # Never remove the first observation.
    available_indices = np.arange(1, n_rows)

    loss_mask = np.zeros(n_rows, dtype=bool)

    lost_count = 0

    # We randomly choose possible burst starting points.
    # Continue until the desired amount of loss is reached.
    while lost_count < target_lost:

        # Find indices that are currently not removed.
        available = available_indices[~loss_mask[1:]]

        if len(available) == 0:
            break

        # Randomly choose the beginning of a burst.
        start = int(rng.choice(available))

        # Remaining observations that can potentially be removed.
        remaining_to_remove = target_lost - lost_count

        # Maximum burst length that can fit inside the trajectory.
        remaining_rows = n_rows - start

        max_allowed_burst = min(
            MAX_BURST_LENGTH,
            remaining_rows,
            remaining_to_remove
        )

        # ----------------------------------------------------
        # IMPORTANT FIX
        # ----------------------------------------------------
        # If only one row is available, rng.integers(2, 2)
        # would cause:
        #
        # ValueError: low >= high
        #
        # Therefore, handle the one-row case separately.
        # ----------------------------------------------------

        if max_allowed_burst <= 0:
            break

        if max_allowed_burst < MIN_BURST_LENGTH:
            burst_length = max_allowed_burst
        else:
            burst_length = int(
                rng.integers(
                    MIN_BURST_LENGTH,
                    max_allowed_burst + 1
                )
            )

        # ----------------------------------------------------
        # Remove consecutive observations.
        # ----------------------------------------------------

        end = min(
            start + burst_length,
            n_rows
        )

        # Only count observations that have not already
        # been removed.
        for idx in range(start, end):

            if not loss_mask[idx]:
                loss_mask[idx] = True
                lost_count += 1

                if lost_count >= target_lost:
                    break

    # Safety rule:
    # Never remove the first observation.
    loss_mask[0] = False

    return loss_mask


# ============================================================
# SIMULATE BURST LOSS
# ============================================================

def simulate_burst_loss(
    input_files,
    target_loss_rate,
    output_dir,
    seed=RANDOM_SEED
):
    """
    Simulate burst packet loss for a collection of trajectories.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    statistics = []

    print()
    print("------------------------------")
    print(
        f"Simulating {int(target_loss_rate * 100)}% burst packet loss"
    )
    print("------------------------------")

    for file_index, input_file in enumerate(input_files, start=1):

        try:
            df = pd.read_csv(input_file)

            if df.empty:
                continue

            n_rows_before = len(df)

            # ------------------------------------------------
            # Create burst-loss mask
            # ------------------------------------------------

            loss_mask = create_burst_loss_mask(
                n_rows=n_rows_before,
                target_loss_rate=target_loss_rate,
                rng=rng
            )

            # ------------------------------------------------
            # Apply loss
            # ------------------------------------------------

            df_received = df.loc[~loss_mask].copy()

            n_rows_after = len(df_received)
            n_rows_lost = n_rows_before - n_rows_after

            # ------------------------------------------------
            # Actual loss rate
            # ------------------------------------------------

            if n_rows_before > 1:
                actual_loss_rate = (
                    n_rows_lost / (n_rows_before - 1)
                )
            else:
                actual_loss_rate = 0.0

            # ------------------------------------------------
            # Find burst lengths
            # ------------------------------------------------

            burst_lengths = []

            current_burst = 0

            for is_lost in loss_mask:

                if is_lost:
                    current_burst += 1

                else:

                    if current_burst > 0:
                        burst_lengths.append(current_burst)

                    current_burst = 0

            # Handle burst reaching the final row.
            if current_burst > 0:
                burst_lengths.append(current_burst)

            number_of_bursts = len(burst_lengths)

            if number_of_bursts > 0:
                average_burst_length = float(
                    np.mean(burst_lengths)
                )

                maximum_burst_length = int(
                    np.max(burst_lengths)
                )
            else:
                average_burst_length = 0.0
                maximum_burst_length = 0

            # ------------------------------------------------
            # Attack observability
            # ------------------------------------------------
            #
            # Important:
            # We calculate this separately from detector
            # performance.
            #
            # If an attack observation is deleted by packet
            # loss, the detector never had access to it.
            # Therefore it is "unobservable", NOT a detector FN.
            # ------------------------------------------------

            original_attack_rows = int(
                (df["label"] != 0).sum()
            )

            received_attack_rows = int(
                (df_received["label"] != 0).sum()
            )

            lost_attack_rows = (
                original_attack_rows -
                received_attack_rows
            )

            if original_attack_rows > 0:
                attack_observability = (
                    received_attack_rows /
                    original_attack_rows
                )
            else:
                attack_observability = np.nan

            # ------------------------------------------------
            # Save simulated trajectory
            # ------------------------------------------------

            output_file = output_dir / input_file.name

            df_received.to_csv(
                output_file,
                index=False
            )

            # ------------------------------------------------
            # Store statistics
            # ------------------------------------------------

            statistics.append(
                {
                    "file": input_file.name,
                    "original_rows": n_rows_before,
                    "received_rows": n_rows_after,
                    "lost_rows": n_rows_lost,
                    "target_loss_rate": target_loss_rate,
                    "actual_loss_rate": actual_loss_rate,
                    "number_of_bursts": number_of_bursts,
                    "average_burst_length": average_burst_length,
                    "maximum_burst_length": maximum_burst_length,
                    "original_attack_rows": original_attack_rows,
                    "received_attack_rows": received_attack_rows,
                    "lost_attack_rows": lost_attack_rows,
                    "attack_observability": attack_observability,
                }
            )

        except Exception as e:

            print(
                f"ERROR processing {input_file.name}: {e}"
            )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics_df = pd.DataFrame(statistics)

    statistics_file = (
        output_dir / "burst_loss_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_file,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    if len(statistics_df) > 0:

        print(
            f"Files processed: {len(statistics_df)}"
        )

        print(
            f"Average actual loss: "
            f"{statistics_df['actual_loss_rate'].mean():.4f}"
        )

        print(
            f"Average rows before: "
            f"{statistics_df['original_rows'].mean():.2f}"
        )

        print(
            f"Average rows after: "
            f"{statistics_df['received_rows'].mean():.2f}"
        )

        print(
            f"Average number of bursts: "
            f"{statistics_df['number_of_bursts'].mean():.2f}"
        )

        print(
            f"Average burst length: "
            f"{statistics_df['average_burst_length'].mean():.2f}"
        )

        print(
            f"Maximum burst length observed: "
            f"{statistics_df['maximum_burst_length'].max()}"
        )

        attack_observability = (
            statistics_df[
                "attack_observability"
            ].dropna()
        )

        if len(attack_observability) > 0:

            print(
                f"Average attack observability: "
                f"{attack_observability.mean():.4f}"
            )

    return statistics_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Find processed trajectories
    # --------------------------------------------------------

    input_files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    print(
        f"Found {len(input_files)} processed trajectories."
    )

    # --------------------------------------------------------
    # Limit to 100 trajectories for testing
    # --------------------------------------------------------

    test_files = input_files[:MAX_TEST_FILES]

    print(
        f"Testing burst-loss simulation on "
        f"{len(test_files)} trajectories."
    )

    # --------------------------------------------------------
    # Run each loss level
    # --------------------------------------------------------

    all_statistics = []

    for loss_rate in LOSS_RATES:

        output_dir = (
            OUTPUT_BASE_DIR /
            f"loss_{int(loss_rate * 100)}"
        )

        stats = simulate_burst_loss(
            input_files=test_files,
            target_loss_rate=loss_rate,
            output_dir=output_dir,
            seed=RANDOM_SEED
        )

        if len(stats) > 0:
            all_statistics.append(stats)

    # --------------------------------------------------------
    # Combine all results
    # --------------------------------------------------------

    if all_statistics:

        combined_statistics = pd.concat(
            all_statistics,
            ignore_index=True
        )

        combined_file = (
            OUTPUT_BASE_DIR /
            "burst_loss_statistics.csv"
        )

        combined_statistics.to_csv(
            combined_file,
            index=False
        )

        print()
        print("====================================")
        print("BURST LOSS SIMULATION COMPLETE")
        print("====================================")

        print(
            f"Total rows in statistics: "
            f"{len(combined_statistics)}"
        )

        print()
        print("Summary by target loss:")

        summary = (
            combined_statistics
            .groupby("target_loss_rate")
            .agg(
                files=("file", "count"),
                avg_original_rows=(
                    "original_rows",
                    "mean"
                ),
                avg_received_rows=(
                    "received_rows",
                    "mean"
                ),
                avg_actual_loss=(
                    "actual_loss_rate",
                    "mean"
                ),
                avg_bursts=(
                    "number_of_bursts",
                    "mean"
                ),
                avg_burst_length=(
                    "average_burst_length",
                    "mean"
                ),
                avg_attack_observability=(
                    "attack_observability",
                    "mean"
                ),
            )
            .reset_index()
        )

        print(summary.to_string(index=False))

        print()
        print(
            f"Statistics saved to:\n"
            f"{combined_file}"
        )
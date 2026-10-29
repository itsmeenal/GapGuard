from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectories"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "simulated"
    / "random"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# EXPERIMENT SETTINGS
# =========================================================

LOSS_RATES = [0.10, 0.30, 0.50]

RANDOM_SEED = 42

# To keep the first run manageable, we initially test
# on a small number of trajectories.
MAX_FILES_FOR_TEST = 100


# =========================================================
# RANDOM GENERATOR
# =========================================================

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================
# FIND TRAJECTORIES
# =========================================================

files = sorted(
    INPUT_DIR.glob("*.csv")
)

print(
    f"Found {len(files)} processed trajectories."
)

test_files = files[:MAX_FILES_FOR_TEST]

print(
    f"Testing random-loss simulation on "
    f"{len(test_files)} trajectories."
)


# =========================================================
# SIMULATION FUNCTION
# =========================================================

def simulate_random_loss(df, loss_rate, rng):

    # ---------------------------------------------
    # Never delete the first observation.
    #
    # It gives us the initial state of the trajectory.
    # ---------------------------------------------

    keep = np.ones(
        len(df),
        dtype=bool
    )

    if len(df) > 1:

        random_values = rng.random(
            len(df) - 1
        )

        keep[1:] = (
            random_values >= loss_rate
        )

    return df.loc[keep].copy()


# =========================================================
# RUN EXPERIMENT
# =========================================================

for loss_rate in LOSS_RATES:

    rate_name = f"loss_{int(loss_rate * 100)}"

    output_rate_dir = (
        OUTPUT_DIR
        / rate_name
    )

    output_rate_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\n------------------------------"
    )

    print(
        f"Simulating {int(loss_rate * 100)}% "
        f"random packet loss"
    )

    print(
        f"------------------------------"
    )


    results = []


    for file_path in test_files:

        df = pd.read_csv(file_path)

        original_rows = len(df)

        simulated = simulate_random_loss(
            df,
            loss_rate,
            rng
        )

        received_rows = len(simulated)

        lost_rows = (
            original_rows
            - received_rows
        )


        # -----------------------------------------
        # Attack observability
        # -----------------------------------------

        original_attack_rows = (
            df["label"] != 0
        ).sum()

        received_attack_rows = (
            simulated["label"] != 0
        ).sum()


        if original_attack_rows > 0:

            attack_observability = (
                received_attack_rows
                / original_attack_rows
            )

        else:

            attack_observability = np.nan


        # -----------------------------------------
        # Save simulated trajectory
        # -----------------------------------------

        output_path = (
            output_rate_dir
            / file_path.name
        )

        simulated.to_csv(
            output_path,
            index=False
        )


        results.append({
            "file": file_path.name,
            "original_rows": original_rows,
            "received_rows": received_rows,
            "lost_rows": lost_rows,
            "actual_loss_fraction": (
                lost_rows / original_rows
            ),
            "original_attack_rows": (
                int(original_attack_rows)
            ),
            "received_attack_rows": (
                int(received_attack_rows)
            ),
            "attack_observability": (
                attack_observability
            ),
        })


    # ---------------------------------------------
    # Save experiment statistics
    # ---------------------------------------------

    results_df = pd.DataFrame(results)

    stats_path = (
        output_rate_dir
        / "loss_statistics.csv"
    )

    results_df.to_csv(
        stats_path,
        index=False
    )


    # ---------------------------------------------
    # Print summary
    # ---------------------------------------------

    print(
        f"Average actual loss: "
        f"{results_df['actual_loss_fraction'].mean():.4f}"
    )

    print(
        f"Average rows before: "
        f"{results_df['original_rows'].mean():.2f}"
    )

    print(
        f"Average rows after: "
        f"{results_df['received_rows'].mean():.2f}"
    )

    attack_files = results_df[
        results_df["original_attack_rows"] > 0
    ]

    if len(attack_files) > 0:

        print(
            f"Average attack observability: "
            f"{attack_files['attack_observability'].mean():.4f}"
        )


print(
    "\n===================================="
)

print(
    "RANDOM LOSS TEST COMPLETE"
)

print(
    "===================================="
)
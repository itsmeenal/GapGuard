from pathlib import Path
import pandas as pd
import numpy as np


INPUT_DIR = Path("data/processed/trajectories")
SIM_DIR = Path("data/simulated/full")

EXPECTED_FILES = 20071

CONDITIONS = {
    "random_10": SIM_DIR / "random" / "loss_10",
    "random_30": SIM_DIR / "random" / "loss_30",
    "random_50": SIM_DIR / "random" / "loss_50",
    "burst_10": SIM_DIR / "burst" / "loss_10",
}


def validate_condition(name, folder):

    print()
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    if not folder.exists():
        print("FAIL — folder does not exist")
        return False

    files = sorted(
        folder.glob("*.csv")
    )

    # loss_statistics.csv is inside the folder,
    # so exclude it from trajectory count.
    files = [
        f for f in files
        if f.name != "loss_statistics.csv"
    ]

    print(
        f"Trajectory files found: {len(files):,}"
    )

    if len(files) != EXPECTED_FILES:
        print(
            f"FAIL — expected {EXPECTED_FILES:,}"
        )
        return False

    stats_path = folder / "loss_statistics.csv"

    if not stats_path.exists():
        print("FAIL — loss_statistics.csv missing")
        return False

    stats = pd.read_csv(stats_path)

    print(
        f"Statistics rows: {len(stats):,}"
    )

    if len(stats) != EXPECTED_FILES:
        print(
            "FAIL — statistics row count does not match"
        )
        return False

    required = [
        "file",
        "loss_type",
        "target_loss_rate",
        "original_rows",
        "received_rows",
        "lost_rows",
        "actual_loss_fraction",
        "original_attack_rows",
        "received_attack_rows",
        "lost_attack_rows",
        "attack_observability",
    ]

    missing = [
        c for c in required
        if c not in stats.columns
    ]

    if missing:
        print(
            "FAIL — missing columns:",
            missing
        )
        return False

    # --------------------------------------------------------
    # Row accounting
    # --------------------------------------------------------

    row_check = (
        stats["original_rows"]
        ==
        stats["received_rows"]
        +
        stats["lost_rows"]
    )

    if not row_check.all():
        print(
            "FAIL — row accounting error"
        )
        return False

    print(
        "PASS — row accounting"
    )

    # --------------------------------------------------------
    # Attack accounting
    # --------------------------------------------------------

    attack_check = (
        stats["original_attack_rows"]
        ==
        stats["received_attack_rows"]
        +
        stats["lost_attack_rows"]
    )

    if not attack_check.all():
        print(
            "FAIL — attack accounting error"
        )
        return False

    print(
        "PASS — attack accounting"
    )

    # --------------------------------------------------------
    # Observability formula
    # --------------------------------------------------------

    valid_attack = (
        stats["original_attack_rows"] > 0
    )

    expected_obs = (
        stats.loc[valid_attack, "received_attack_rows"]
        /
        stats.loc[valid_attack, "original_attack_rows"]
    )

    actual_obs = (
        stats.loc[valid_attack, "attack_observability"]
    )

    if not np.allclose(
        expected_obs,
        actual_obs,
        atol=1e-10
    ):
        print(
            "FAIL — attack observability formula"
        )
        return False

    print(
        "PASS — attack observability formula"
    )

    # --------------------------------------------------------
    # Loss rate
    # --------------------------------------------------------

    mean_loss = stats[
        "actual_loss_fraction"
    ].mean()

    target = stats[
        "target_loss_rate"
    ].iloc[0]

    print(
        f"Target loss      : {target:.2%}"
    )

    print(
        f"Mean actual loss : {mean_loss:.2%}"
    )

    if abs(mean_loss - target) > 0.005:
        print(
            "FAIL — mean loss differs from target"
        )
        return False

    print(
        "PASS — loss rate is within tolerance"
    )

    # --------------------------------------------------------
    # File-name uniqueness
    # --------------------------------------------------------

    if stats["file"].nunique() != EXPECTED_FILES:
        print(
            "FAIL — duplicate filenames in statistics"
        )
        return False

    print(
        "PASS — unique trajectory filenames"
    )

    # --------------------------------------------------------
    # Basic output integrity
    # --------------------------------------------------------

    sample_files = files[:20]

    for path in sample_files:

        df = pd.read_csv(path)

        if len(df) == 0:
            print(
                f"FAIL — empty trajectory: {path.name}"
            )
            return False

        if not np.isfinite(
            df.select_dtypes(
                include=[np.number]
            ).to_numpy()
        ).all():
            print(
                f"FAIL — non-finite values: {path.name}"
            )
            return False

    print(
        "PASS — sampled output files are valid"
    )

    print(
        "CONDITION PASSED"
    )

    return True


def main():

    print("=" * 60)
    print("STEP 2.8 — FINAL STAGE-2 VALIDATION")
    print("=" * 60)

    original_files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    print(
        f"Original processed trajectories: "
        f"{len(original_files):,}"
    )

    if len(original_files) != EXPECTED_FILES:
        print(
            "FAIL — original trajectory count changed"
        )
        return

    print(
        "PASS — original trajectory count preserved"
    )

    results = {}

    for name, folder in CONDITIONS.items():

        results[name] = validate_condition(
            name,
            folder
        )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    for name, passed in results.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{name:<15} {status}"
        )

    if all(results.values()):

        print()
        print(
            "STAGE 2 CORE VALIDATION PASSED"
        )

        print()
        print(
            "Note:"
        )

        print(
            "Burst 30% and Burst 50% are intentionally "
            "not stored as full datasets."
        )

        print(
            "They will be generated deterministically "
            "during later experiments."
        )

    else:

        print()
        print(
            "STAGE 2 VALIDATION FAILED"
        )


if __name__ == "__main__":
    main()
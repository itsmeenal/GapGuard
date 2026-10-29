from pathlib import Path
import json


RESULTS = Path("results")


CONDITIONS = [
    "clean",
    "random_10",
    "random_30",
    "random_50",
    "burst_10",
    "burst_30",
    "burst_50",
]


# Frozen calibration results from Stage 3.9.
# These are deliberately written explicitly so the
# robustness summary is reproducible and auditable.

RESULTS_TABLE = {
    "clean": {
        "global_f1": 0.0924,
        "gapguard_f1": 0.1054,
        "global_fpr": 0.0103,
        "gapguard_fpr": 0.0103,
        "global_recall": 0.0539,
        "gapguard_recall": 0.0619,
    },
    "random_10": {
        "global_f1": 0.0993,
        "gapguard_f1": 0.1082,
        "global_fpr": 0.0157,
        "gapguard_fpr": 0.0124,
        "global_recall": 0.0613,
        "gapguard_recall": 0.0649,
    },
    "random_30": {
        "global_f1": 0.1111,
        "gapguard_f1": 0.1160,
        "global_fpr": 0.0301,
        "gapguard_fpr": 0.0207,
        "global_recall": 0.0782,
        "gapguard_recall": 0.0755,
    },
    "random_50": {
        "global_f1": 0.1267,
        "gapguard_f1": 0.1296,
        "global_fpr": 0.0534,
        "gapguard_fpr": 0.0377,
        "global_recall": 0.1073,
        "gapguard_recall": 0.0980,
    },
    "burst_10": {
        "global_f1": 0.0958,
        "gapguard_f1": 0.1075,
        "global_fpr": 0.0144,
        "gapguard_fpr": 0.0133,
        "global_recall": 0.0583,
        "gapguard_recall": 0.0651,
    },
    "burst_30": {
        "global_f1": 0.1038,
        "gapguard_f1": 0.1109,
        "global_fpr": 0.0258,
        "gapguard_fpr": 0.0218,
        "global_recall": 0.0702,
        "gapguard_recall": 0.0727,
    },
    "burst_50": {
        "global_f1": 0.1117,
        "gapguard_f1": 0.1161,
        "global_fpr": 0.0449,
        "gapguard_fpr": 0.0375,
        "global_recall": 0.0887,
        "gapguard_recall": 0.0873,
    },
}


def relative_change(old, new):
    if old == 0:
        return None
    return (new - old) / old * 100.0


def main():

    print("=" * 80)
    print("STAGE 4.4 — ROBUSTNESS SUMMARY")
    print("=" * 80)

    f1_improvements = []
    fpr_reductions = []

    rows = []

    for condition in CONDITIONS:

        result = RESULTS_TABLE[condition]

        f1_change = relative_change(
            result["global_f1"],
            result["gapguard_f1"],
        )

        fpr_change = relative_change(
            result["global_fpr"],
            result["gapguard_fpr"],
        )

        if f1_change is not None:
            f1_improvements.append(f1_change)

        if fpr_change is not None:
            fpr_reductions.append(-fpr_change)

        rows.append({
            "condition": condition,
            **result,
            "f1_relative_change_percent": f1_change,
            "fpr_reduction_percent": -fpr_change,
        })

    f1_positive = sum(
        row["gapguard_f1"] > row["global_f1"]
        for row in rows
    )

    fpr_reduced = sum(
        row["gapguard_fpr"] < row["global_fpr"]
        for row in rows
    )

    packet_rows = [
        row for row in rows
        if row["condition"] != "clean"
    ]

    packet_fpr_reduced = sum(
        row["gapguard_fpr"] < row["global_fpr"]
        for row in packet_rows
    )

    average_f1_improvement = (
        sum(f1_improvements) / len(f1_improvements)
    )

    average_packet_fpr_reduction = (
        sum(
            row["fpr_reduction_percent"]
            for row in packet_rows
        )
        / len(packet_rows)
    )

    print("\nCondition-level results")
    print("-" * 80)

    print(
        f"{'Condition':15s}"
        f"{'Global F1':>12s}"
        f"{'GapGuard F1':>14s}"
        f"{'Global FPR':>13s}"
        f"{'GapGuard FPR':>14s}"
    )

    for row in rows:

        print(
            f"{row['condition']:15s}"
            f"{row['global_f1']:12.4f}"
            f"{row['gapguard_f1']:14.4f}"
            f"{row['global_fpr'] * 100:12.2f}%"
            f"{row['gapguard_fpr'] * 100:13.2f}%"
        )

    print("\nConsistency")
    print("-" * 80)

    print(
        f"F1 improved in : "
        f"{f1_positive}/{len(rows)} conditions"
    )

    print(
        f"FPR reduced in packet-loss conditions : "
        f"{packet_fpr_reduced}/{len(packet_rows)}"
    )

    print(
        f"Average relative F1 improvement : "
        f"{average_f1_improvement:.2f}%"
    )

    print(
        f"Average packet-loss FPR reduction : "
        f"{average_packet_fpr_reduction:.2f}%"
    )

    output = {
        "stage": "4.4",
        "purpose": (
            "Condition-level robustness and consistency "
            "summary for frozen calibration"
        ),
        "results": rows,
        "summary": {
            "f1_improved_conditions": f1_positive,
            "total_conditions": len(rows),
            "packet_loss_fpr_reduced_conditions": packet_fpr_reduced,
            "packet_loss_conditions": len(packet_rows),
            "average_relative_f1_improvement_percent":
                average_f1_improvement,
            "average_packet_loss_fpr_reduction_percent":
                average_packet_fpr_reduction,
        },
        "interpretation": {
            "f1_consistency":
                "GapGuard improves F1 in every evaluated condition.",
            "fpr_consistency":
                "GapGuard reduces FPR in every evaluated packet-loss condition.",
            "statistical_significance":
                "Not tested because paired per-flight predictions were not saved."
        }
    }

    output_dir = RESULTS / "robustness"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        "robustness_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2
        )

    print(f"\nSaved: {output_path}")
    print("\nSTAGE 4.4 COMPLETE")


if __name__ == "__main__":
    main()
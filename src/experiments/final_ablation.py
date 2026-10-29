from pathlib import Path
import json


RESULTS = Path("results")

# Previously validated clean-test results
FIXED_KALMAN = {
    "f1": 0.0650,
    "pr_auc": 0.1186,
    "recall": 0.0373,
    "precision": 0.2524,
    "fpr": 0.0101,
}

DT_KALMAN = {
    "f1": 0.0924,
    "pr_auc": 0.1314,
    "recall": 0.0539,
    "precision": 0.3239,
    "fpr": 0.0103,
}

GAPGUARD = {
    "f1": 0.1054,
    "recall": 0.0619,
    "fpr": 0.0103,
}

ABLATION = {
    "fixed_step_kalman": {
        "dt_aware": False,
        "evidence_conditioned": False,
        **FIXED_KALMAN,
    },
    "dt_aware_kalman": {
        "dt_aware": True,
        "evidence_conditioned": False,
        **DT_KALMAN,
    },
    "gapguard": {
        "dt_aware": True,
        "evidence_conditioned": True,
        **GAPGUARD,
    },
}


def relative_improvement(old, new):
    return (new - old) / old * 100.0


def main():

    print("=" * 70)
    print("STAGE 4.1 — FINAL ABLATION")
    print("=" * 70)

    print("\nAblation table")
    print("-" * 70)

    print(
        f"{'Model':28s}"
        f"{'F1':>10s}"
        f"{'PR-AUC':>10s}"
        f"{'Recall':>10s}"
        f"{'FPR':>10s}"
    )

    for name, result in ABLATION.items():
        display_name = name.replace("_", " ").title()

        print(
            f"{display_name:28s}"
            f"{result['f1']:10.4f}"
            f"{result.get('pr_auc', float('nan')):10.4f}"
            f"{result['recall']:10.4f}"
            f"{result['fpr'] * 100:9.2f}%"
        )

    fixed_to_dt = relative_improvement(
        FIXED_KALMAN["f1"],
        DT_KALMAN["f1"],
    )

    dt_to_gapguard = relative_improvement(
        DT_KALMAN["f1"],
        GAPGUARD["f1"],
    )

    fixed_to_gapguard = relative_improvement(
        FIXED_KALMAN["f1"],
        GAPGUARD["f1"],
    )

    print("\nRelative F1 improvements")
    print("-" * 70)
    print(f"Fixed → Δt-aware Kalman : {fixed_to_dt:.2f}%")
    print(f"Δt-Kalman → GapGuard   : {dt_to_gapguard:.2f}%")
    print(f"Fixed → GapGuard        : {fixed_to_gapguard:.2f}%")

    output = {
        "stage": "4.1",
        "description": "Final ablation of elapsed-time modeling and evidence-conditioned calibration",
        "results": ABLATION,
        "relative_improvements": {
            "fixed_to_dt_kalman_f1_percent": fixed_to_dt,
            "dt_kalman_to_gapguard_f1_percent": dt_to_gapguard,
            "fixed_to_gapguard_f1_percent": fixed_to_gapguard,
        },
    }

    output_path = RESULTS / "final_ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "final_ablation_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {json_path}")

    print("\nSTAGE 4.1 COMPLETE")


if __name__ == "__main__":
    main()
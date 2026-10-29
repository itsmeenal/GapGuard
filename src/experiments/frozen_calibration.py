import json
from pathlib import Path

import numpy as np

from src.data.experiment_loader import load_manifest

# Reuse the already validated Stage 3.8 implementation.
from src.experiments.evidence_conformal import (
    collect,
    fit_evidence_boundaries,
    evidence_bins,
    fit_global_threshold,
    fit_evidence_thresholds,
    metrics,
    fpr_by_evidence,
)


# ============================================================
# CONFIGURATION
# ============================================================

CONDITIONS = [
    ("clean", "random", 0.00),

    ("random_10", "random", 0.10),
    ("random_30", "random", 0.30),
    ("random_50", "random", 0.50),

    ("burst_10", "burst", 0.10),
    ("burst_30", "burst", 0.30),
    ("burst_50", "burst", 0.50),
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        "STAGE 3.9 — "
        "FROZEN CLEAN-CALIBRATION ROBUSTNESS"
    )
    print("=" * 78)

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    manifest = load_manifest()

    val_manifest = (
        manifest[
            manifest["split"] == "val"
        ]
        .reset_index(drop=True)
    )

    test_manifest = (
        manifest[
            manifest["split"] == "test"
        ]
        .reset_index(drop=True)
    )

    # ========================================================
    # STEP 1:
    # CALIBRATE ON CLEAN VALIDATION DATA ONLY
    # ========================================================

    print(
        "\nSTEP 1 — CLEAN VALIDATION CALIBRATION ONLY"
    )

    print(
        "\nCollecting clean calibration data..."
    )

    clean_calibration = collect(
        val_manifest,
        "random",
        0.00,
    )

    # --------------------------------------------------------
    # Evidence boundaries
    # --------------------------------------------------------

    boundaries = fit_evidence_boundaries(
        clean_calibration
    )

    calibration_bins = evidence_bins(
        clean_calibration["gaps"],
        clean_calibration["uncertainty"],
        boundaries,
    )

    # --------------------------------------------------------
    # Global threshold
    # --------------------------------------------------------

    global_threshold = fit_global_threshold(
        clean_calibration
    )

    # --------------------------------------------------------
    # Evidence-conditioned thresholds
    # --------------------------------------------------------

    evidence_thresholds = fit_evidence_thresholds(
        clean_calibration,
        calibration_bins,
    )

    print(
        "\nCLEAN CALIBRATION FROZEN"
    )

    print(
        f"Global threshold : "
        f"{global_threshold:.6f}"
    )

    print(
        "\nEvidence boundaries:"
    )

    print(
        f"  gap q1 : "
        f"{boundaries['gap_q1']:.6f}"
    )

    print(
        f"  gap q2 : "
        f"{boundaries['gap_q2']:.6f}"
    )

    print(
        f"  unc q1 : "
        f"{boundaries['unc_q1']:.6f}"
    )

    print(
        f"  unc q2 : "
        f"{boundaries['unc_q2']:.6f}"
    )

    print(
        "\nEvidence thresholds:"
    )

    print(
        f"  low    : "
        f"{evidence_thresholds[0]:.6f}"
    )

    print(
        f"  medium : "
        f"{evidence_thresholds[1]:.6f}"
    )

    print(
        f"  high   : "
        f"{evidence_thresholds[2]:.6f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These values will NOT be changed "
        "for any packet-loss condition."
    )

    # ========================================================
    # STEP 2:
    # TEST ALL CONDITIONS WITH FROZEN CALIBRATION
    # ========================================================

    results = []

    for (
        condition_name,
        loss_type,
        loss_rate,
    ) in CONDITIONS:

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"TEST CONDITION: {condition_name}"
        )

        print(
            "=" * 78
        )

        # ----------------------------------------------------
        # Collect test data
        # ----------------------------------------------------

        test = collect(
            test_manifest,
            loss_type,
            loss_rate,
        )

        scores = test["scores"]
        labels = test["labels"]

        # ----------------------------------------------------
        # Assign evidence bins using CLEAN boundaries only
        # ----------------------------------------------------

        test_bins = evidence_bins(
            test["gaps"],
            test["uncertainty"],
            boundaries,
        )

        # ====================================================
        # MODEL A:
        # FROZEN GLOBAL THRESHOLD
        # ====================================================

        global_predictions = (
            scores >= global_threshold
        ).astype(np.int64)

        global_metrics = metrics(
            labels,
            scores,
            global_predictions,
        )

        global_fpr_bins = fpr_by_evidence(
            labels,
            global_predictions,
            test_bins,
        )

        # ====================================================
        # MODEL B:
        # FROZEN EVIDENCE-CONDITIONED THRESHOLDS
        # ====================================================

        conditioned_predictions = np.zeros(
            len(scores),
            dtype=np.int64,
        )

        for evidence in [0, 1, 2]:

            selected = (
                test_bins == evidence
            )

            threshold = (
                evidence_thresholds[
                    evidence
                ]
            )

            conditioned_predictions[
                selected
            ] = (
                scores[selected]
                >= threshold
            ).astype(np.int64)

        conditioned_metrics = metrics(
            labels,
            scores,
            conditioned_predictions,
        )

        conditioned_fpr_bins = fpr_by_evidence(
            labels,
            conditioned_predictions,
            test_bins,
        )

        # ====================================================
        # EVIDENCE DISTRIBUTION
        # ====================================================

        evidence_distribution = {}

        names = {
            0: "low",
            1: "medium",
            2: "high",
        }

        for evidence in [0, 1, 2]:

            count = int(
                (
                    test_bins
                    == evidence
                ).sum()
            )

            evidence_distribution[
                names[evidence]
            ] = {
                "rows": count,
                "fraction": float(
                    count
                    / len(test_bins)
                ),
            }

        # ====================================================
        # PRINT
        # ====================================================

        print(
            "\nFROZEN GLOBAL"
        )

        print(
            f"  Precision : "
            f"{global_metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{global_metrics['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{global_metrics['f1']:.4f}"
        )

        print(
            f"  PR-AUC    : "
            f"{global_metrics['pr_auc']:.4f}"
        )

        print(
            f"  FPR       : "
            f"{global_metrics['fpr']:.4f}"
        )

        print(
            "\nFROZEN EVIDENCE-CONDITIONED"
        )

        print(
            f"  Precision : "
            f"{conditioned_metrics['precision']:.4f}"
        )

        print(
            f"  Recall    : "
            f"{conditioned_metrics['recall']:.4f}"
        )

        print(
            f"  F1        : "
            f"{conditioned_metrics['f1']:.4f}"
        )

        print(
            f"  PR-AUC    : "
            f"{conditioned_metrics['pr_auc']:.4f}"
        )

        print(
            f"  FPR       : "
            f"{conditioned_metrics['fpr']:.4f}"
        )

        # ----------------------------------------------------
        # FPR by evidence
        # ----------------------------------------------------

        print(
            "\nFPR BY EVIDENCE"
        )

        for name in [
            "low",
            "medium",
            "high",
        ]:

            global_fpr = (
                global_fpr_bins[
                    name
                ]["fpr"]
            )

            conditioned_fpr = (
                conditioned_fpr_bins[
                    name
                ]["fpr"]
            )

            print(
                f"  {name:<7} "
                f"global="
                f"{global_fpr} "
                f"conditioned="
                f"{conditioned_fpr}"
            )

        # ----------------------------------------------------
        # Evidence distribution
        # ----------------------------------------------------

        print(
            "\nEVIDENCE DISTRIBUTION"
        )

        for name in [
            "low",
            "medium",
            "high",
        ]:

            info = (
                evidence_distribution[
                    name
                ]
            )

            print(
                f"  {name:<7}: "
                f"{info['fraction']:.4f}"
            )

        # ====================================================
        # STORE RESULT
        # ====================================================

        results.append({

            "condition":
                condition_name,

            "loss_type":
                loss_type,

            "loss_rate":
                loss_rate,

            "global_metrics":
                global_metrics,

            "conditioned_metrics":
                conditioned_metrics,

            "global_fpr_by_evidence":
                global_fpr_bins,

            "conditioned_fpr_by_evidence":
                conditioned_fpr_bins,

            "evidence_distribution":
                evidence_distribution,
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = Path(
        "results/frozen_calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "frozen_clean_calibration_results.json"
    )

    output = {

        "calibration_source":
            "clean_validation_only",

        "global_threshold":
            float(global_threshold),

        "evidence_boundaries":
            boundaries,

        "evidence_thresholds": {
            str(k): float(v)
            for k, v
            in evidence_thresholds.items()
        },

        "results":
            results,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print(
        "\n\n"
        + "=" * 105
    )

    print(
        "FINAL FROZEN-CALIBRATION ROBUSTNESS"
    )

    print(
        "=" * 105
    )

    print(
        f"{'Condition':<15}"
        f"{'Global F1':>12}"
        f"{'Cond F1':>12}"
        f"{'Global FPR':>14}"
        f"{'Cond FPR':>12}"
        f"{'Global Rec':>14}"
        f"{'Cond Rec':>12}"
    )

    print(
        "-" * 105
    )

    for result in results:

        gm = (
            result[
                "global_metrics"
            ]
        )

        cm = (
            result[
                "conditioned_metrics"
            ]
        )

        print(
            f"{result['condition']:<15}"
            f"{gm['f1']:>12.4f}"
            f"{cm['f1']:>12.4f}"
            f"{gm['fpr']:>14.4f}"
            f"{cm['fpr']:>12.4f}"
            f"{gm['recall']:>14.4f}"
            f"{cm['recall']:>12.4f}"
        )

    print(
        "\nFrozen calibration source:"
    )

    print(
        "  CLEAN validation data only"
    )

    print(
        "\nResult saved to:"
    )

    print(
        output_file.resolve()
    )

    print(
        "\nSTAGE 3.9 COMPLETE"
    )


if __name__ == "__main__":
    main()
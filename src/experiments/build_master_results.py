from pathlib import Path
import json
import csv


RESULTS = Path("results")


def load_json(path):
    if not path.exists():
        print(f"[WARNING] Missing: {path}")
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 80)
    print("STAGE 4.2 — MASTER EXPERIMENT SUMMARY")
    print("=" * 80)

    master = {
        "project": "GapGuard",
        "description": (
            "Gap-aware anomaly detection for streaming ADS-B telemetry "
            "under packet loss"
        ),
        "experiments": {}
    }

    # ---------------------------------------------------------
    # 1. Isolation Forest
    # ---------------------------------------------------------

    if_result = load_json(
        RESULTS / "baseline_isolation_forest" /
        "clean_test_result.json"
    )

    if if_result:
        master["experiments"]["isolation_forest"] = if_result

    # ---------------------------------------------------------
    # 2. Kalman comparison
    # ---------------------------------------------------------

    kalman_result = load_json(
        RESULTS / "baseline_kalman" /
        "clean_test_comparison.json"
    )

    if kalman_result:
        master["experiments"]["kalman_comparison"] = kalman_result

    # ---------------------------------------------------------
    # 3. Standalone GRU
    # ---------------------------------------------------------

    gru_result = load_json(
        RESULTS / "neural_residual" /
        "clean_test_result.json"
    )

    if gru_result:
        master["experiments"]["standalone_gru"] = gru_result

    # ---------------------------------------------------------
    # 4. Kalman + GRU
    # ---------------------------------------------------------

    hybrid_result = load_json(
        RESULTS / "hybrid_kalman_gru" /
        "clean_test_result.json"
    )

    if hybrid_result:
        master["experiments"]["kalman_gru_hybrid"] = hybrid_result

    # ---------------------------------------------------------
    # 5. Packet loss
    # ---------------------------------------------------------

    packet_result = load_json(
        RESULTS / "packet_loss" /
        "kalman_packet_loss_results.json"
    )

    if packet_result:
        master["experiments"]["packet_loss"] = packet_result

    # ---------------------------------------------------------
    # 6. Evidence-conditioned conformal
    # ---------------------------------------------------------

    conformal_result = load_json(
        RESULTS / "evidence_conformal" /
        "evidence_conformal_results.json"
    )

    if conformal_result:
        master["experiments"]["evidence_conformal"] = conformal_result

    # ---------------------------------------------------------
    # 7. Frozen calibration
    # ---------------------------------------------------------

    frozen_result = load_json(
        RESULTS / "frozen_calibration" /
        "frozen_clean_calibration_results.json"
    )

    if frozen_result:
        master["experiments"]["frozen_calibration"] = frozen_result

    # ---------------------------------------------------------
    # 8. Final ablation
    # ---------------------------------------------------------

    ablation_result = load_json(
        RESULTS / "final_ablation" /
        "final_ablation_results.json"
    )

    if ablation_result:
        master["experiments"]["final_ablation"] = ablation_result

    # ---------------------------------------------------------
    # Save master JSON
    # ---------------------------------------------------------

    output_dir = RESULTS / "master"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "master_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    # ---------------------------------------------------------
    # Create compact summary CSV
    # ---------------------------------------------------------

    rows = []

    def add_row(
        experiment,
        condition,
        f1=None,
        precision=None,
        recall=None,
        fpr=None,
        pr_auc=None,
        roc_auc=None
    ):
        rows.append({
            "experiment": experiment,
            "condition": condition,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
        })

    # Fixed / Delta-t Kalman
    if kalman_result:

        fixed = kalman_result.get("fixed_step", {})
        dt = kalman_result.get("delta_t_aware", {})

        if fixed:
            add_row(
                "Fixed-step Kalman",
                "clean",
                fixed.get("f1"),
                fixed.get("precision"),
                fixed.get("recall"),
                fixed.get("fpr"),
                fixed.get("pr_auc"),
                fixed.get("roc_auc"),
            )

        if dt:
            add_row(
                "Delta-t Kalman",
                "clean",
                dt.get("f1"),
                dt.get("precision"),
                dt.get("recall"),
                dt.get("fpr"),
                dt.get("pr_auc"),
                dt.get("roc_auc"),
            )

    # Isolation Forest
    if if_result:
        add_row(
            "Isolation Forest",
            "clean",
            if_result.get("f1"),
            if_result.get("precision"),
            if_result.get("recall"),
            if_result.get("fpr"),
            if_result.get("pr_auc"),
            if_result.get("roc_auc"),
        )

    # Standalone GRU
    if gru_result:
        add_row(
            "Standalone GRU",
            "clean",
            gru_result.get("f1"),
            gru_result.get("precision"),
            gru_result.get("recall"),
            gru_result.get("fpr"),
            gru_result.get("pr_auc"),
            gru_result.get("roc_auc"),
        )

    # Hybrid
    if hybrid_result:
        add_row(
            "Delta-t Kalman + GRU",
            "clean",
            hybrid_result.get("f1"),
            hybrid_result.get("precision"),
            hybrid_result.get("recall"),
            hybrid_result.get("fpr"),
            hybrid_result.get("pr_auc"),
            hybrid_result.get("roc_auc"),
        )

    # Final GapGuard clean result
    if frozen_result:

        clean = None

        if isinstance(frozen_result, dict):
            conditions = frozen_result.get("conditions", [])

            if isinstance(conditions, list):
                for item in conditions:
                    if item.get("condition") == "clean":
                        clean = item
                        break

        if clean:
            add_row(
                "GapGuard",
                "clean",
                clean.get("conditioned_f1"),
                None,
                clean.get("conditioned_recall"),
                clean.get("conditioned_fpr"),
                None,
                None,
            )

    csv_path = output_dir / "master_metrics.csv"

    fieldnames = [
        "experiment",
        "condition",
        "f1",
        "precision",
        "recall",
        "fpr",
        "pr_auc",
        "roc_auc",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\nExperiments included:")
    for name in master["experiments"]:
        print(f"  ✓ {name}")

    print(f"\nMaster JSON : {json_path}")
    print(f"Master CSV  : {csv_path}")

    print("\nSTAGE 4.2 COMPLETE")


if __name__ == "__main__":
    main()
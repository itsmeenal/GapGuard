from pathlib import Path
import json


RESULTS = Path("results")


REQUIRED_FILES = {
    "Isolation Forest":
        RESULTS / "baseline_isolation_forest" /
        "clean_test_result.json",

    "Kalman comparison":
        RESULTS / "baseline_kalman" /
        "clean_test_comparison.json",

    "Standalone GRU":
        RESULTS / "neural_residual" /
        "clean_test_result.json",

    "Kalman-GRU hybrid":
        RESULTS / "hybrid_kalman_gru" /
        "clean_test_result.json",

    "Packet-loss experiment":
        RESULTS / "packet_loss" /
        "kalman_packet_loss_results.json",

    "Evidence-conditioned calibration":
        RESULTS / "evidence_conformal" /
        "evidence_conformal_results.json",

    "Frozen calibration":
        RESULTS / "frozen_calibration" /
        "frozen_clean_calibration_results.json",

    "Final ablation":
        RESULTS / "final_ablation" /
        "final_ablation_results.json",

    "Robustness summary":
        RESULTS / "robustness" /
        "robustness_summary.json",

    "Master results":
        RESULTS / "master" /
        "master_results.json",

    "Master metrics":
        RESULTS / "master" /
        "master_metrics.csv",

    "Figure 1 PNG":
        RESULTS / "figures" /
        "figure1_baseline_comparison.png",

    "Figure 1 PDF":
        RESULTS / "figures" /
        "figure1_baseline_comparison.pdf",

    "Figure 2 PNG":
        RESULTS / "figures" /
        "figure2_packet_loss_fpr.png",

    "Figure 2 PDF":
        RESULTS / "figures" /
        "figure2_packet_loss_fpr.pdf",

    "Figure 3 PNG":
        RESULTS / "figures" /
        "figure3_evidence_stratified_fpr.png",

    "Figure 3 PDF":
        RESULTS / "figures" /
        "figure3_evidence_stratified_fpr.pdf",
}


def main():

    print("=" * 80)
    print("STAGE 4.5 — FINAL REPRODUCIBILITY VALIDATION")
    print("=" * 80)

    passed = 0
    failed = 0

    for name, path in REQUIRED_FILES.items():

        if path.exists():
            size_kb = path.stat().st_size / 1024

            print(
                f"[PASS] {name:35s}"
                f"{size_kb:10.1f} KB"
            )

            passed += 1

        else:
            print(
                f"[FAIL] {name:35s}"
                f"MISSING"
            )

            failed += 1

    print("\n" + "-" * 80)

    print(f"Required artifacts : {len(REQUIRED_FILES)}")
    print(f"Passed             : {passed}")
    print(f"Failed             : {failed}")

    summary = {
        "stage": "4.5",
        "project": "GapGuard",
        "required_artifacts": len(REQUIRED_FILES),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "INCOMPLETE",
    }

    output_dir = RESULTS / "validation"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir /
        "final_validation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"\nSaved: {output_path}")

    if failed == 0:
        print("\nSTAGE 4.5 COMPLETE — ALL ARTIFACTS VERIFIED")
    else:
        print("\nSTAGE 4.5 INCOMPLETE — MISSING ARTIFACTS")


if __name__ == "__main__":
    main()
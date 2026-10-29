from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_isolation_forest"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "data")
)

from src.data.experiment_loader import (
    FEATURE_COLUMNS,
    load_split,
    load_trajectory,
    get_split_manifest,
    apply_loss,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# Keep this moderate so the experiment finishes quickly.
MAX_TRAIN_ROWS = 100_000

CONTAMINATION = "auto"


# ============================================================
# HELPERS
# ============================================================

def collect_rows(
    split,
    normal_only=False,
    max_rows=None,
):
    """
    Collect trajectory rows from one split.
    """

    manifest = get_split_manifest(split)

    X_parts = []
    y_parts = []

    rows_collected = 0

    for _, row in manifest.iterrows():

        X, y, _ = load_trajectory(row)

        if normal_only:

            mask = (
                y == 0
            )

            X = X[mask]
            y = y[mask]

        if len(X) == 0:
            continue

        if (
            max_rows is not None
            and rows_collected + len(X)
            > max_rows
        ):

            remaining = (
                max_rows - rows_collected
            )

            if remaining <= 0:
                break

            X = X[:remaining]
            y = y[:remaining]

        X_parts.append(X)
        y_parts.append(y)

        rows_collected += len(X)

        if (
            max_rows is not None
            and rows_collected >= max_rows
        ):
            break

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X, y


def collect_test_with_loss(
    loss_rate=0.0,
    loss_type="none",
):
    """
    Collect the complete test set.

    Packet loss is applied trajectory-by-trajectory
    so trajectory boundaries are preserved.
    """

    manifest = get_split_manifest(
        "test"
    )

    X_parts = []
    y_parts = []

    original_attack_rows = 0
    received_attack_rows = 0

    for _, row in manifest.iterrows():

        X, y, _ = load_trajectory(row)

        original_attack_rows += np.sum(
            y != 0
        )

        if loss_type == "none":

            received_X = X
            received_y = y

        else:

            identifier = (
                f"{row['route']}_"
                f"{row['flight_id']}_"
                f"{row['file_label']}"
            )

            (
                received_X,
                received_y,
                _
            ) = apply_loss(
                X,
                y,
                loss_rate,
                loss_type,
                identifier,
            )

        received_attack_rows += np.sum(
            received_y != 0
        )

        if len(received_X) > 0:

            X_parts.append(
                received_X
            )

            y_parts.append(
                received_y
            )

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return (
        X,
        y,
        original_attack_rows,
        received_attack_rows,
    )


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold(
    scores
):
    """
    Choose anomaly-score threshold using
    normal validation observations.

    We use the 99th percentile of normal
    validation anomaly scores.

    This gives approximately 1% validation
    false-positive rate.

    IMPORTANT:
    Test labels are NOT used here.
    """

    threshold = np.percentile(
        scores,
        99
    )

    return float(threshold)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 3.3 — ISOLATION FOREST BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nLoading normal training data...")

    X_train, y_train = collect_rows(
        split="train",
        normal_only=True,
        max_rows=MAX_TRAIN_ROWS,
    )

    print(
        f"Training rows : "
        f"{len(X_train):,}"
    )

    print(
        f"Training attack rows : "
        f"{np.sum(y_train != 0):,}"
    )

    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    print("\nFitting feature scaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    print(
        "\nTraining Isolation Forest..."
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled
    )

    print(
        "Isolation Forest trained."
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nLoading normal validation data..."
    )

    X_val, y_val = collect_rows(
        split="val",
        normal_only=True,
        max_rows=50_000,
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    # sklearn:
    # decision_function higher = more normal
    #
    # Therefore we negate it so:
    # higher score = more anomalous.

    val_scores = -model.decision_function(
        X_val_scaled
    )

    threshold = choose_threshold(
        val_scores
    )

    print(
        f"Validation rows : "
        f"{len(X_val):,}"
    )

    print(
        f"Selected anomaly threshold : "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating on clean test data..."
    )

    (
        X_test,
        y_test,
        original_attack_rows,
        received_attack_rows,
    ) = collect_test_with_loss(
        loss_rate=0.0,
        loss_type="none",
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    test_scores = -model.decision_function(
        X_test_scaled
    )

    predictions = (
        test_scores >= threshold
    ).astype(int)

    true_labels = (
        y_test != 0
    ).astype(int)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            true_labels,
            test_scores,
        )
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(
            true_labels,
            test_scores,
        )
    except ValueError:
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("ISOLATION FOREST — CLEAN TEST")
    print("=" * 60)

    print(
        f"Test rows       : {len(y_test):,}"
    )

    print(
        f"Attack rows     : "
        f"{np.sum(true_labels):,}"
    )

    print(
        f"Predicted alerts: "
        f"{np.sum(predictions):,}"
    )

    print(
        f"\nPrecision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"FPR       : {fpr:.4f}"
    )

    print(
        "\nConfusion matrix:"
    )

    print(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    result = {
        "model": "IsolationForest",
        "loss_type": "none",
        "loss_rate": 0.0,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "attack_rows": int(np.sum(true_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    output_file = (
        RESULTS_DIR
        / "clean_test_result.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"\nResult saved to:\n"
        f"{output_file}"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "STAGE 3.3 COMPLETE"
    )

    print(
        "=" * 60
    )
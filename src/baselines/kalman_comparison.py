"""
Stage 3.4 — Fixed-step vs Delta-t-aware Kalman baseline

Compares:
    1. Fixed-step Kalman
    2. Delta-t-aware Kalman

Binary evaluation:
    label 0       -> normal
    labels 1,2,3  -> attack

Run from GapGuard root:

    python -m src.baselines.kalman_comparison
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data.experiment_loader import (
    load_manifest,
    load_trajectory,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_kalman"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Fixed-step baseline assumes every observation arrives
# approximately every 5 seconds.
FIXED_DT = 5.0

# Safety limits for actual elapsed time.
MIN_DT = 0.1
MAX_DT = 120.0

# Kalman state:
#
# [latitude,
#  longitude,
#  altitude,
#  speed]
#
STATE_DIM = 4


# ============================================================
# MEASUREMENT SCALING
# ============================================================
#
# The four variables have very different numerical scales:
#
# latitude       ~ 10-20
# longitude      ~ 110-130
# altitude       ~ hundreds/thousands
# speed          ~ tens/hundreds
#
# Without scaling, altitude can dominate the innovation score.
#
# These are fixed engineering scales, not learned from labels.
# ============================================================

MEASUREMENT_SCALE = np.array([
    1.0,       # latitude
    1.0,       # longitude
    100.0,     # altitude meters
    50.0,      # speed km/h
])


# Process noise in standardized measurement space.

PROCESS_NOISE = np.array([
    0.01,
    0.01,
    0.10,
    0.10,
])


# Measurement noise in standardized measurement space.

MEASUREMENT_NOISE = np.array([
    0.01,
    0.01,
    0.10,
    0.10,
])


# ============================================================
# DATAFRAME HANDLING
# ============================================================

def get_dataframe(loader_output):
    """
    Normalize the output of load_trajectory().

    Depending on the loader implementation, it may return:

        DataFrame

    or:

        (DataFrame, something)

    This function extracts the actual pandas DataFrame.
    """

    # Case 1:
    # load_trajectory() directly returned a DataFrame.

    if hasattr(
        loader_output,
        "columns"
    ):

        return loader_output


    # Case 2:
    # load_trajectory() returned a tuple/list.

    if isinstance(
        loader_output,
        (tuple, list)
    ):

        for item in loader_output:

            if hasattr(
                item,
                "columns"
            ):

                return item


    raise TypeError(
        "load_trajectory() did not return "
        "a pandas DataFrame or a tuple containing "
        "a pandas DataFrame. "
        f"Received type: {type(loader_output)}"
    )


# ============================================================
# KALMAN FILTER
# ============================================================

class SimpleKalman:

    def __init__(
        self,
        use_actual_dt=True
    ):

        self.use_actual_dt = (
            use_actual_dt
        )

        self.x = None
        self.P = None

        self.R = np.diag(
            MEASUREMENT_NOISE
        )


    # --------------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------------

    def initialize(
        self,
        measurement
    ):

        self.x = np.asarray(
            measurement,
            dtype=float
        ).copy()

        self.P = np.eye(
            STATE_DIM
        )

        # Initial uncertainty.

        self.P[0, 0] = 0.01
        self.P[1, 1] = 0.01
        self.P[2, 2] = 1.0
        self.P[3, 3] = 1.0


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(
        self,
        dt
    ):

        # ----------------------------------------------------
        # Fixed-step version
        # ----------------------------------------------------

        if self.use_actual_dt:

            effective_dt = np.clip(
                dt,
                MIN_DT,
                MAX_DT
            )

        else:

            effective_dt = FIXED_DT


        # Identity state transition.
        #
        # We deliberately keep the physical model simple.
        # The experiment is testing whether uncertainty should
        # depend on actual elapsed time.

        F = np.eye(
            STATE_DIM
        )


        # ----------------------------------------------------
        # Process uncertainty
        # ----------------------------------------------------

        q_scale = max(
            effective_dt / FIXED_DT,
            0.1
        )

        Q = np.diag(
            PROCESS_NOISE
            * q_scale
        )


        # Prediction.

        self.x = (
            F @ self.x
        )

        self.P = (
            F
            @ self.P
            @ F.T
            + Q
        )


    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        measurement
    ):

        z = np.asarray(
            measurement,
            dtype=float
        )

        H = np.eye(
            STATE_DIM
        )


        # Innovation.

        innovation = (
            z
            - H @ self.x
        )


        # Innovation covariance.

        S = (
            H
            @ self.P
            @ H.T
            + self.R
        )


        # ----------------------------------------------------
        # Mahalanobis innovation score
        # ----------------------------------------------------

        try:

            score = float(
                innovation.T
                @ np.linalg.solve(
                    S,
                    innovation
                )
            )

        except np.linalg.LinAlgError:

            score = float(
                innovation.T
                @ np.linalg.pinv(S)
                @ innovation
            )


        # ----------------------------------------------------
        # Kalman gain
        # ----------------------------------------------------

        try:

            K = (
                self.P
                @ H.T
                @ np.linalg.inv(S)
            )

        except np.linalg.LinAlgError:

            K = (
                self.P
                @ H.T
                @ np.linalg.pinv(S)
            )


        # State update.

        self.x = (
            self.x
            + K @ innovation
        )


        # ----------------------------------------------------
        # Joseph-form covariance update
        # ----------------------------------------------------

        I = np.eye(
            STATE_DIM
        )

        self.P = (
            (I - K @ H)
            @ self.P
            @ (I - K @ H).T
            + K
            @ self.R
            @ K.T
        )


        return score


# ============================================================
# EXTRACT MEASUREMENTS
# ============================================================

def get_measurements(
    df
):

    raw = df[
        [
            "latitude",
            "longitude",
            "altitude_meters",
            "speed_kmh",
        ]
    ].to_numpy(
        dtype=float
    )


    # --------------------------------------------------------
    # Scale variables to comparable ranges.
    # --------------------------------------------------------

    scaled = (
        raw
        / MEASUREMENT_SCALE
    )

    return scaled


# ============================================================
# SCORE ONE TRAJECTORY
# ============================================================

def score_trajectory(
    df,
    use_actual_dt
):

    # Make sure we have a DataFrame.

    df = get_dataframe(
        df
    )


    # Measurements.

    measurements = (
        get_measurements(
            df
        )
    )


    # --------------------------------------------------------
    # BINARY LABEL CONVERSION
    # --------------------------------------------------------
    #
    # Dataset labels:
    #
    # 0 = normal
    # 1 = attack type 1
    # 2 = attack type 2
    # 3 = attack type 3
    #
    # Detector evaluation:
    #
    # 0 = normal
    # 1 = any attack
    # --------------------------------------------------------

    raw_labels = df[
        "label"
    ].to_numpy(
        dtype=int
    )

    labels = (
        raw_labels > 0
    ).astype(int)


    # Actual timestamps.

    times = df[
        "t_elapsed_sec"
    ].to_numpy(
        dtype=float
    )


    n = len(
        measurements
    )


    scores = np.zeros(
        n,
        dtype=float
    )


    if n == 0:

        return (
            scores,
            labels
        )


    # --------------------------------------------------------
    # Initialize Kalman filter.
    # --------------------------------------------------------

    kf = SimpleKalman(
        use_actual_dt=
        use_actual_dt
    )


    kf.initialize(
        measurements[0]
    )


    # First point has no previous observation.

    scores[0] = 0.0


    # --------------------------------------------------------
    # Process remaining observations.
    # --------------------------------------------------------

    for i in range(
        1,
        n
    ):

        dt = (
            times[i]
            - times[i - 1]
        )


        # Protect against malformed timestamps.

        if (
            not np.isfinite(dt)
            or dt <= 0
        ):

            dt = FIXED_DT


        # Prediction.

        kf.predict(
            dt
        )


        # Measurement update + anomaly score.

        scores[i] = (
            kf.update(
                measurements[i]
            )
        )


    return (
        scores,
        labels
    )


# ============================================================
# SCORE ALL TRAJECTORIES
# ============================================================

def collect_scores(
    manifest,
    split_name,
    use_actual_dt
):

    split = manifest[
        manifest["split"]
        == split_name
    ]


    all_scores = []
    all_labels = []


    print(
        f"Scoring {split_name}: "
        f"{len(split):,} trajectories..."
    )


    # iterrows() returns a pandas Series.
    # load_trajectory() expects row["route"], etc.

    for idx, (_, row) in enumerate(
        split.iterrows(),
        start=1
    ):

        loader_output = (
            load_trajectory(
                row
            )
        )


        df = get_dataframe(
            loader_output
        )


        scores, labels = (
            score_trajectory(
                df,
                use_actual_dt
            )
        )


        all_scores.append(
            scores
        )

        all_labels.append(
            labels
        )


        if idx % 500 == 0:

            print(
                f"  processed "
                f"{idx:,}/"
                f"{len(split):,}"
            )


    if not all_scores:

        return (
            np.array(
                [],
                dtype=float
            ),
            np.array(
                [],
                dtype=int
            )
        )


    scores = np.concatenate(
        all_scores
    )

    labels = np.concatenate(
        all_labels
    )


    return (
        scores,
        labels
    )


# ============================================================
# THRESHOLD
# ============================================================

def select_threshold(
    scores,
    labels
):

    # Only NORMAL validation observations
    # are allowed to determine the threshold.

    normal_scores = (
        scores[
            labels == 0
        ]
    )


    if len(
        normal_scores
    ) == 0:

        raise ValueError(
            "No normal validation observations "
            "were found for threshold calibration."
        )


    # 99th percentile corresponds approximately
    # to 1% validation false-positive rate.

    threshold = float(
        np.percentile(
            normal_scores,
            99
        )
    )


    return threshold


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    scores,
    labels,
    threshold
):

    # Binary predictions.

    predictions = (
        scores >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Standard classification metrics
    # --------------------------------------------------------

    precision = precision_score(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )


    recall = recall_score(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )


    f1 = f1_score(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )


    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            labels,
            scores
        )

    except ValueError:

        roc_auc = None


    try:

        pr_auc = (
            average_precision_score(
                labels,
                scores
            )
        )

    except ValueError:

        pr_auc = None


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        cm.ravel()
    )


    # --------------------------------------------------------
    # False-positive rate
    # --------------------------------------------------------

    if (
        fp + tn
    ) > 0:

        fpr = (
            fp
            / (fp + tn)
        )

    else:

        fpr = 0.0


    return {

        "threshold":
            float(threshold),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            (
                None
                if roc_auc is None
                else float(roc_auc)
            ),

        "pr_auc":
            (
                None
                if pr_auc is None
                else float(pr_auc)
            ),

        "fpr":
            float(fpr),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "test_rows":
            int(len(labels)),

        "attack_rows":
            int(labels.sum()),

        "predicted_alerts":
            int(predictions.sum()),
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_result(
    name,
    result
):

    print()

    print(
        "=" * 60
    )

    print(
        name
    )

    print(
        "=" * 60
    )


    print(
        f"Test rows       : "
        f"{result['test_rows']:,}"
    )


    print(
        f"Attack rows     : "
        f"{result['attack_rows']:,}"
    )


    print(
        f"Predicted alerts: "
        f"{result['predicted_alerts']:,}"
    )


    print()


    print(
        f"Precision : "
        f"{result['precision']:.4f}"
    )


    print(
        f"Recall    : "
        f"{result['recall']:.4f}"
    )


    print(
        f"F1        : "
        f"{result['f1']:.4f}"
    )


    if (
        result["roc_auc"]
        is not None
    ):

        print(
            f"ROC-AUC   : "
            f"{result['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC   : N/A"
        )


    if (
        result["pr_auc"]
        is not None
    ):

        print(
            f"PR-AUC    : "
            f"{result['pr_auc']:.4f}"
        )

    else:

        print(
            "PR-AUC    : N/A"
        )


    print(
        f"FPR       : "
        f"{result['fpr']:.4f}"
    )


    print()


    print(
        "Confusion matrix:"
    )


    print(
        f"TN={result['tn']}, "
        f"FP={result['fp']}, "
        f"FN={result['fn']}, "
        f"TP={result['tp']}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "STAGE 3.4 — "
        "FIXED-STEP vs DELTA-T KALMAN"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    manifest = load_manifest()


    print()


    print(
        f"Manifest rows : "
        f"{len(manifest):,}"
    )


    # ========================================================
    # VALIDATION — FIXED STEP
    # ========================================================

    print()


    print(
        "Loading normal validation data "
        "for fixed-step Kalman..."
    )


    (
        val_scores_fixed,
        val_labels_fixed
    ) = collect_scores(
        manifest,
        "val",
        False
    )


    threshold_fixed = (
        select_threshold(
            val_scores_fixed,
            val_labels_fixed
        )
    )


    print(
        f"Fixed-step threshold : "
        f"{threshold_fixed:.6f}"
    )


    # ========================================================
    # VALIDATION — DELTA-T
    # ========================================================

    print()


    print(
        "Loading normal validation data "
        "for delta-t-aware Kalman..."
    )


    (
        val_scores_dt,
        val_labels_dt
    ) = collect_scores(
        manifest,
        "val",
        True
    )


    threshold_dt = (
        select_threshold(
            val_scores_dt,
            val_labels_dt
        )
    )


    print(
        f"Delta-t threshold    : "
        f"{threshold_dt:.6f}"
    )


    # ========================================================
    # TEST — FIXED STEP
    # ========================================================

    print()


    print(
        "Evaluating fixed-step Kalman..."
    )


    (
        test_scores_fixed,
        test_labels_fixed
    ) = collect_scores(
        manifest,
        "test",
        False
    )


    fixed_result = evaluate(
        test_scores_fixed,
        test_labels_fixed,
        threshold_fixed
    )


    print_result(
        "FIXED-STEP KALMAN — CLEAN TEST",
        fixed_result
    )


    # ========================================================
    # TEST — DELTA-T
    # ========================================================

    print()


    print(
        "Evaluating delta-t-aware Kalman..."
    )


    (
        test_scores_dt,
        test_labels_dt
    ) = collect_scores(
        manifest,
        "test",
        True
    )


    dt_result = evaluate(
        test_scores_dt,
        test_labels_dt,
        threshold_dt
    )


    print_result(
        "DELTA-T-AWARE KALMAN — CLEAN TEST",
        dt_result
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {

        "experiment":
            "fixed_step_vs_delta_t_kalman",

        "fixed_step":
            fixed_result,

        "delta_t_aware":
            dt_result,

        "fixed_dt_seconds":
            FIXED_DT,

        "measurement_features":
            [
                "latitude",
                "longitude",
                "altitude_meters",
                "speed_kmh"
            ],

        "label_mapping":
            {
                "0":
                    "normal",

                "1":
                    "attack",

                "2":
                    "attack",

                "3":
                    "attack"
            },

        "note":
            "Labels are used only for evaluation. "
            "Derived columns and label columns are "
            "not detector inputs."
    }


    output_path = (
        RESULT_DIR
        / "clean_test_comparison.json"
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


    print()


    print(
        "=" * 60
    )


    print(
        "STAGE 3.4 COMPLETE"
    )


    print(
        "=" * 60
    )


    print()


    print(
        "Result saved to:"
    )


    print(
        output_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
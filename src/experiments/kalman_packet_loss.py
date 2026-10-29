import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

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
    make_loss_mask,
)


# ============================================================
# CONFIGURATION
# ============================================================

FIXED_DT = 5.0

MIN_DT = 0.1
MAX_DT = 120.0

MEASUREMENT_COLUMNS = [
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
]

# Scale the measurements so the Mahalanobis score is numerically stable.
MEASUREMENT_SCALE = np.array(
    [1.0, 1.0, 100.0, 50.0],
    dtype=np.float64,
)

R = np.diag([
    0.01,
    0.01,
    0.10,
    0.10,
])

Q_BASE = np.diag([
    0.01,
    0.01,
    0.10,
    0.10,
])


# ============================================================
# KALMAN FILTER
# ============================================================

class SimpleKalman:

    def __init__(self, use_delta_t=True):
        self.use_delta_t = use_delta_t
        self.state = None
        self.P = np.eye(4)

    def update(self, z, dt):

        z = np.asarray(
            z,
            dtype=np.float64,
        )

        # First observation initializes the filter.
        if self.state is None:
            self.state = z.copy()
            return 0.0

        # ----------------------------------------------------
        # Fixed-step vs actual elapsed-time
        # ----------------------------------------------------

        if self.use_delta_t:

            effective_dt = np.clip(
                dt,
                MIN_DT,
                MAX_DT,
            )

        else:

            effective_dt = FIXED_DT

        # Simple identity transition.
        F = np.eye(4)

        # Process uncertainty grows with elapsed time.
        Q = (
            Q_BASE
            * effective_dt
            / FIXED_DT
        )

        # Prediction.
        predicted = F @ self.state

        predicted_P = (
            F
            @ self.P
            @ F.T
            + Q
        )

        # Innovation.
        innovation = (
            z
            - predicted
        )

        # Innovation covariance.
        S = (
            predicted_P
            + R
        )

        try:

            S_inv = np.linalg.inv(S)

        except np.linalg.LinAlgError:

            S_inv = np.linalg.pinv(S)

        # Mahalanobis innovation score.
        score = float(
            innovation.T
            @ S_inv
            @ innovation
        )

        # Kalman gain.
        K = (
            predicted_P
            @ S_inv
        )

        # Update state.
        self.state = (
            predicted
            + K @ innovation
        )

        # Update covariance.
        self.P = (
            np.eye(4)
            - K
        ) @ predicted_P

        return score


# ============================================================
# HELPERS
# ============================================================

def get_dataframe(output):

    if isinstance(
        output,
        pd.DataFrame,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):

        for item in output:

            if isinstance(
                item,
                pd.DataFrame,
            ):
                return item

    raise TypeError(
        "Could not extract DataFrame from load_trajectory()."
    )


def deterministic_seed(identifier):

    """
    Convert a trajectory identifier into a stable
    integer seed.

    This avoids relying on Python's built-in hash(),
    whose value can vary between processes.
    """

    digest = hashlib.sha256(
        identifier.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# ============================================================
# LOSS APPLICATION
# ============================================================

def apply_loss_to_dataframe(
    df,
    row,
    loss_type,
    loss_rate,
):

    # No loss.
    if loss_rate == 0:

        return (
            df.copy()
            .reset_index(drop=True)
        )

    identifier = (
        f"{row['route']}_"
        f"{row['flight_id']}_"
        f"{row['file_label']}"
    )

    seed = deterministic_seed(
        identifier
        + "|"
        + loss_type
        + "|"
        + str(loss_rate)
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # make_loss_mask() in the existing experiment loader
    # does NOT accept seed= or seed_identifier=.
    #
    # We therefore create the deterministic RNG here and
    # reproduce the random/burst mask directly.
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    n_rows = len(df)

    keep = np.ones(
        n_rows,
        dtype=bool,
    )

    # Always retain first observation.
    if n_rows <= 1:

        return (
            df.copy()
            .reset_index(drop=True)
        )

    target_loss = int(
        round(
            loss_rate
            * (n_rows - 1)
        )
    )

    if target_loss <= 0:

        return (
            df.copy()
            .reset_index(drop=True)
        )

    # ========================================================
    # RANDOM LOSS
    # ========================================================

    if loss_type == "random":

        candidates = np.arange(
            1,
            n_rows,
        )

        lost_indices = rng.choice(
            candidates,
            size=min(
                target_loss,
                len(candidates),
            ),
            replace=False,
        )

        keep[lost_indices] = False

    # ========================================================
    # BURST LOSS
    # ========================================================

    elif loss_type == "burst":

        lost = 0

        while lost < target_loss:

            # Random burst length between 2 and 8.
            burst_length = int(
                rng.integers(
                    2,
                    9,
                )
            )

            # Only start from observations after first.
            start = int(
                rng.integers(
                    1,
                    n_rows,
                )
            )

            end = min(
                start + burst_length,
                n_rows,
            )

            before = int(
                (~keep[start:end]).sum()
            )

            keep[start:end] = False

            after = int(
                (~keep[start:end]).sum()
            )

            newly_lost = (
                after - before
            )

            lost += newly_lost

            # Safety condition.
            if (
                (~keep[1:]).sum()
                >= target_loss
            ):
                break

        # If overlapping bursts produced too many lost rows,
        # restore some observations so the requested loss rate
        # is respected approximately.
        current_lost = np.flatnonzero(
            ~keep
        )

        if len(current_lost) > target_loss:

            restore_count = (
                len(current_lost)
                - target_loss
            )

            restore = rng.choice(
                current_lost,
                size=restore_count,
                replace=False,
            )

            keep[restore] = True

    else:

        raise ValueError(
            f"Unknown loss_type: {loss_type}"
        )

    # Safety: first observation is always retained.
    keep[0] = True

    return (
        df.iloc[keep]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# TRAJECTORY SCORING
# ============================================================

def score_trajectory(
    df,
    use_delta_t,
):

    values = (
        df[
            MEASUREMENT_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
        / MEASUREMENT_SCALE
    )

    times = (
        df[
            "t_elapsed_sec"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    labels = (
        df[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
        > 0
    ).astype(
        np.int64
    )

    kalman = SimpleKalman(
        use_delta_t=use_delta_t
    )

    scores = []

    for i in range(len(df)):

        if i == 0:

            dt = FIXED_DT

        else:

            dt = (
                times[i]
                - times[i - 1]
            )

        score = kalman.update(
            values[i],
            dt,
        )

        scores.append(score)

    return (
        np.asarray(scores),
        labels,
    )


# ============================================================
# COLLECT SCORES
# ============================================================

def collect_scores(
    manifest,
    loss_type,
    loss_rate,
    use_delta_t,
):

    all_scores = []
    all_labels = []

    original_attack = 0
    surviving_attack = 0

    original_rows = 0
    surviving_rows = 0

    for idx, (_, row) in enumerate(
        manifest.iterrows()
    ):

        loaded = load_trajectory(
            row
        )

        df = get_dataframe(
            loaded
        )

        # Original labels before packet loss.
        original_labels = (
            df["label"]
            .to_numpy()
            > 0
        )

        original_attack += int(
            original_labels.sum()
        )

        original_rows += len(df)

        # Apply simulated packet loss.
        df_loss = (
            apply_loss_to_dataframe(
                df,
                row,
                loss_type,
                loss_rate,
            )
        )

        # Surviving labels.
        surviving_labels = (
            df_loss["label"]
            .to_numpy()
            > 0
        )

        surviving_attack += int(
            surviving_labels.sum()
        )

        surviving_rows += len(
            df_loss
        )

        # Score surviving observations.
        scores, labels = (
            score_trajectory(
                df_loss,
                use_delta_t,
            )
        )

        all_scores.append(
            scores
        )

        all_labels.append(
            labels
        )

        if (
            (idx + 1) % 500
            == 0
        ):

            print(
                f"    processed "
                f"{idx + 1:,}/"
                f"{len(manifest):,}"
            )

    scores = np.concatenate(
        all_scores
    )

    labels = np.concatenate(
        all_labels
    )

    observability = (
        surviving_attack
        / original_attack
        if original_attack > 0
        else 0.0
    )

    return {

        "scores": scores,

        "labels": labels,

        "original_rows":
            original_rows,

        "surviving_rows":
            surviving_rows,

        "original_attack":
            original_attack,

        "surviving_attack":
            surviving_attack,

        "observability":
            observability,
    }


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_threshold(
    scores,
    labels,
):

    normal_scores = scores[
        labels == 0
    ]

    return float(
        np.percentile(
            normal_scores,
            99,
        )
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    scores,
    labels,
    threshold,
):

    predictions = (
        scores >= threshold
    ).astype(
        np.int64
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        labels,
        scores,
    )

    pr_auc = average_precision_score(
        labels,
        scores,
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

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

        "alerts":
            int(predictions.sum()),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print(
        "STAGE 3.7 — "
        "PACKET-LOSS KALMAN EXPERIMENT"
    )
    print("=" * 65)

    manifest = load_manifest()

    val_manifest = (
        manifest[
            manifest["split"]
            == "val"
        ]
        .reset_index(
            drop=True
        )
    )

    test_manifest = (
        manifest[
            manifest["split"]
            == "test"
        ]
        .reset_index(
            drop=True
        )
    )

    conditions = [

        (
            "clean",
            "random",
            0.00,
        ),

        (
            "random_10",
            "random",
            0.10,
        ),

        (
            "random_30",
            "random",
            0.30,
        ),

        (
            "random_50",
            "random",
            0.50,
        ),

        (
            "burst_10",
            "burst",
            0.10,
        ),

        (
            "burst_30",
            "burst",
            0.30,
        ),

        (
            "burst_50",
            "burst",
            0.50,
        ),
    ]

    results = []

    # ========================================================
    # RUN CONDITIONS
    # ========================================================

    for (
        condition_name,
        loss_type,
        loss_rate,
    ) in conditions:

        print(
            "\n"
            + "=" * 65
        )

        print(
            f"CONDITION: "
            f"{condition_name}"
        )

        print(
            "=" * 65
        )

        # ----------------------------------------------------
        # FIXED-STEP VALIDATION
        # ----------------------------------------------------

        print(
            "\n  Fixed-step validation..."
        )

        fixed_val = collect_scores(
            val_manifest,
            loss_type,
            loss_rate,
            use_delta_t=False,
        )

        fixed_threshold = (
            select_threshold(
                fixed_val["scores"],
                fixed_val["labels"],
            )
        )

        print(
            f"  Fixed threshold: "
            f"{fixed_threshold:.6f}"
        )

        # ----------------------------------------------------
        # DELTA-T VALIDATION
        # ----------------------------------------------------

        print(
            "  Δt-aware validation..."
        )

        delta_val = collect_scores(
            val_manifest,
            loss_type,
            loss_rate,
            use_delta_t=True,
        )

        delta_threshold = (
            select_threshold(
                delta_val["scores"],
                delta_val["labels"],
            )
        )

        print(
            f"  Δt threshold: "
            f"{delta_threshold:.6f}"
        )

        # ----------------------------------------------------
        # FIXED-STEP TEST
        # ----------------------------------------------------

        print(
            "\n  Fixed-step test..."
        )

        fixed_test = collect_scores(
            test_manifest,
            loss_type,
            loss_rate,
            use_delta_t=False,
        )

        fixed_metrics = evaluate(
            fixed_test["scores"],
            fixed_test["labels"],
            fixed_threshold,
        )

        # ----------------------------------------------------
        # DELTA-T TEST
        # ----------------------------------------------------

        print(
            "  Δt-aware test..."
        )

        delta_test = collect_scores(
            test_manifest,
            loss_type,
            loss_rate,
            use_delta_t=True,
        )

        delta_metrics = evaluate(
            delta_test["scores"],
            delta_test["labels"],
            delta_threshold,
        )

        # ----------------------------------------------------
        # SAVE FIXED RESULT
        # ----------------------------------------------------

        fixed_result = {

            "condition":
                condition_name,

            "model":
                "fixed_step_kalman",

            "loss_type":
                loss_type,

            "loss_rate":
                loss_rate,

            "observability":
                fixed_test[
                    "observability"
                ],

            "original_rows":
                fixed_test[
                    "original_rows"
                ],

            "surviving_rows":
                fixed_test[
                    "surviving_rows"
                ],

            "original_attack":
                fixed_test[
                    "original_attack"
                ],

            "surviving_attack":
                fixed_test[
                    "surviving_attack"
                ],

            **fixed_metrics,
        }

        # ----------------------------------------------------
        # SAVE DELTA-T RESULT
        # ----------------------------------------------------

        delta_result = {

            "condition":
                condition_name,

            "model":
                "delta_t_kalman",

            "loss_type":
                loss_type,

            "loss_rate":
                loss_rate,

            "observability":
                delta_test[
                    "observability"
                ],

            "original_rows":
                delta_test[
                    "original_rows"
                ],

            "surviving_rows":
                delta_test[
                    "surviving_rows"
                ],

            "original_attack":
                delta_test[
                    "original_attack"
                ],

            "surviving_attack":
                delta_test[
                    "surviving_attack"
                ],

            **delta_metrics,
        }

        results.append(
            fixed_result
        )

        results.append(
            delta_result
        )

        # ----------------------------------------------------
        # PRINT CONDITION RESULT
        # ----------------------------------------------------

        print(
            "\n  Attack observability: "
            f"{delta_test['observability']:.4f}"
        )

        print(
            "\n  FIXED-STEP"
        )

        print(
            f"    F1       : "
            f"{fixed_metrics['f1']:.4f}"
        )

        print(
            f"    PR-AUC   : "
            f"{fixed_metrics['pr_auc']:.4f}"
        )

        print(
            f"    Recall   : "
            f"{fixed_metrics['recall']:.4f}"
        )

        print(
            f"    FPR      : "
            f"{fixed_metrics['fpr']:.4f}"
        )

        print(
            "\n  DELTA-T"
        )

        print(
            f"    F1       : "
            f"{delta_metrics['f1']:.4f}"
        )

        print(
            f"    PR-AUC   : "
            f"{delta_metrics['pr_auc']:.4f}"
        )

        print(
            f"    Recall   : "
            f"{delta_metrics['recall']:.4f}"
        )

        print(
            f"    FPR      : "
            f"{delta_metrics['fpr']:.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_dir = Path(
        "results/packet_loss"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "kalman_packet_loss_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "FINAL PACKET-LOSS COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"{'Condition':<15}"
        f"{'Model':<20}"
        f"{'Obs.':>8}"
        f"{'F1':>8}"
        f"{'PR-AUC':>10}"
        f"{'Recall':>10}"
        f"{'FPR':>10}"
    )

    print(
        "-" * 90
    )

    for r in results:

        print(
            f"{r['condition']:<15}"
            f"{r['model']:<20}"
            f"{r['observability']:>8.3f}"
            f"{r['f1']:>8.3f}"
            f"{r['pr_auc']:>10.3f}"
            f"{r['recall']:>10.3f}"
            f"{r['fpr']:>10.3f}"
        )

    print(
        "\nResult saved to:"
    )

    print(
        output_file.resolve()
    )

    print(
        "\nSTAGE 3.7 COMPLETE"
    )


if __name__ == "__main__":
    main()
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data.experiment_loader import (
    load_manifest,
    load_trajectory,
)


# ============================================================
# CONFIG
# ============================================================

FIXED_DT = 5.0
MIN_DT = 0.1
MAX_DT = 120.0

ALPHA = 0.01

MEASUREMENT_COLUMNS = [
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
]

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
# DELTA-T KALMAN
# ============================================================

class DeltaTKalman:

    def __init__(self):
        self.state = None
        self.P = np.eye(4)

    def update(self, z, dt):

        z = np.asarray(
            z,
            dtype=np.float64,
        )

        if self.state is None:
            self.state = z.copy()

            return (
                0.0,
                float(np.trace(self.P)),
            )

        effective_dt = np.clip(
            dt,
            MIN_DT,
            MAX_DT,
        )

        F = np.eye(4)

        Q = (
            Q_BASE
            * effective_dt
            / FIXED_DT
        )

        predicted = (
            F @ self.state
        )

        predicted_P = (
            F
            @ self.P
            @ F.T
            + Q
        )

        innovation = (
            z - predicted
        )

        S = (
            predicted_P
            + R
        )

        try:
            S_inv = np.linalg.inv(S)

        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        severity = float(
            innovation.T
            @ S_inv
            @ innovation
        )

        # Uncertainty BEFORE seeing current observation.
        uncertainty = float(
            np.trace(predicted_P)
        )

        K = (
            predicted_P
            @ S_inv
        )

        self.state = (
            predicted
            + K @ innovation
        )

        self.P = (
            np.eye(4)
            - K
        ) @ predicted_P

        return (
            severity,
            uncertainty,
        )


# ============================================================
# DATA HELPERS
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
        "Could not extract DataFrame."
    )


def deterministic_seed(identifier):

    digest = hashlib.sha256(
        identifier.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


# ============================================================
# PACKET LOSS
# ============================================================

def apply_loss(
    df,
    row,
    loss_type,
    loss_rate,
):

    if loss_rate == 0:

        return (
            df.copy()
            .reset_index(drop=True)
        )

    identifier = (
        f"{row['route']}_"
        f"{row['flight_id']}_"
        f"{row['file_label']}_"
        f"{loss_type}_"
        f"{loss_rate}"
    )

    seed = deterministic_seed(
        identifier
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(df)

    keep = np.ones(
        n,
        dtype=bool,
    )

    if n <= 1:
        return df.copy()

    target = int(
        round(
            loss_rate
            * (n - 1)
        )
    )

    if target <= 0:
        return df.copy()

    # --------------------------------------------------------
    # RANDOM LOSS
    # --------------------------------------------------------

    if loss_type == "random":

        candidates = np.arange(
            1,
            n,
        )

        lost = rng.choice(
            candidates,
            size=min(
                target,
                len(candidates),
            ),
            replace=False,
        )

        keep[lost] = False

    # --------------------------------------------------------
    # BURST LOSS
    # --------------------------------------------------------

    elif loss_type == "burst":

        while (
            (~keep[1:]).sum()
            < target
        ):

            burst_length = int(
                rng.integers(
                    2,
                    9,
                )
            )

            start = int(
                rng.integers(
                    1,
                    n,
                )
            )

            end = min(
                start + burst_length,
                n,
            )

            keep[
                start:end
            ] = False

        current_lost = (
            np.flatnonzero(
                ~keep
            )
        )

        if len(current_lost) > target:

            restore_count = (
                len(current_lost)
                - target
            )

            restore = rng.choice(
                current_lost,
                size=restore_count,
                replace=False,
            )

            keep[restore] = True

    else:

        raise ValueError(
            f"Unknown loss type: {loss_type}"
        )

    keep[0] = True

    return (
        df.iloc[keep]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# SCORE TRAJECTORY
# ============================================================

def score_trajectory(df):

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
        df["label"]
        .to_numpy(
            dtype=np.int64
        )
        > 0
    ).astype(
        np.int64
    )

    model = DeltaTKalman()

    severity = []
    gaps = []
    uncertainty = []

    for i in range(len(df)):

        if i == 0:

            dt = FIXED_DT

        else:

            dt = (
                times[i]
                - times[i - 1]
            )

        score, unc = (
            model.update(
                values[i],
                dt,
            )
        )

        severity.append(
            score
        )

        gaps.append(
            dt
        )

        uncertainty.append(
            unc
        )

    return (
        np.asarray(
            severity,
            dtype=np.float64,
        ),
        np.asarray(
            gaps,
            dtype=np.float64,
        ),
        np.asarray(
            uncertainty,
            dtype=np.float64,
        ),
        labels,
    )


# ============================================================
# COLLECT
# ============================================================

def collect(
    manifest,
    loss_type,
    loss_rate,
):

    scores = []
    gaps = []
    uncertainty = []
    labels = []

    for idx, (_, row) in enumerate(
        manifest.iterrows()
    ):

        df = get_dataframe(
            load_trajectory(
                row
            )
        )

        df = apply_loss(
            df,
            row,
            loss_type,
            loss_rate,
        )

        (
            s,
            g,
            u,
            y,
        ) = score_trajectory(
            df
        )

        scores.append(s)
        gaps.append(g)
        uncertainty.append(u)
        labels.append(y)

        if (
            (idx + 1) % 500
            == 0
        ):

            print(
                f"    processed "
                f"{idx + 1:,}/"
                f"{len(manifest):,}"
            )

    return {
        "scores":
            np.concatenate(scores),

        "gaps":
            np.concatenate(gaps),

        "uncertainty":
            np.concatenate(
                uncertainty
            ),

        "labels":
            np.concatenate(labels),
    }


# ============================================================
# EVIDENCE SCORE
# ============================================================

def fit_evidence_boundaries(
    calibration,
):

    normal = (
        calibration[
            "labels"
        ] == 0
    )

    normal_gaps = (
        calibration[
            "gaps"
        ][normal]
    )

    normal_uncertainty = (
        calibration[
            "uncertainty"
        ][normal]
    )

    # Boundaries learned ONLY from validation normal data.
    gap_q1, gap_q2 = (
        np.quantile(
            normal_gaps,
            [1 / 3, 2 / 3],
        )
    )

    unc_q1, unc_q2 = (
        np.quantile(
            normal_uncertainty,
            [1 / 3, 2 / 3],
        )
    )

    return {
        "gap_q1":
            float(gap_q1),

        "gap_q2":
            float(gap_q2),

        "unc_q1":
            float(unc_q1),

        "unc_q2":
            float(unc_q2),
    }


def evidence_bins(
    gaps,
    uncertainty,
    boundaries,
):

    # Higher gap / uncertainty means
    # LOWER evidence quality.

    gap_level = np.zeros(
        len(gaps),
        dtype=np.int64,
    )

    gap_level[
        gaps
        > boundaries[
            "gap_q1"
        ]
    ] = 1

    gap_level[
        gaps
        > boundaries[
            "gap_q2"
        ]
    ] = 2

    unc_level = np.zeros(
        len(uncertainty),
        dtype=np.int64,
    )

    unc_level[
        uncertainty
        > boundaries[
            "unc_q1"
        ]
    ] = 1

    unc_level[
        uncertainty
        > boundaries[
            "unc_q2"
        ]
    ] = 2

    # Conservative combination:
    # worst evidence indicator wins.
    risk_level = np.maximum(
        gap_level,
        unc_level,
    )

    # Convert risk to evidence:
    #
    # 0 = low evidence
    # 1 = medium evidence
    # 2 = high evidence

    evidence = (
        2 - risk_level
    )

    return evidence


# ============================================================
# CONFORMAL THRESHOLD
# ============================================================

def conformal_quantile(
    values,
    alpha=ALPHA,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return np.inf

    sorted_values = np.sort(
        values
    )

    n = len(
        sorted_values
    )

    # Split-conformal upper quantile.
    rank = int(
        np.ceil(
            (n + 1)
            * (1 - alpha)
        )
    )

    rank = min(
        max(
            rank,
            1,
        ),
        n,
    )

    return float(
        sorted_values[
            rank - 1
        ]
    )


def fit_global_threshold(
    calibration,
):

    normal_scores = (
        calibration[
            "scores"
        ][
            calibration[
                "labels"
            ] == 0
        ]
    )

    return conformal_quantile(
        normal_scores,
        ALPHA,
    )


def fit_evidence_thresholds(
    calibration,
    calibration_bins,
):

    thresholds = {}

    normal = (
        calibration[
            "labels"
        ] == 0
    )

    global_threshold = (
        conformal_quantile(
            calibration[
                "scores"
            ][normal],
            ALPHA,
        )
    )

    for evidence in [
        0,
        1,
        2,
    ]:

        selected = (
            normal
            & (
                calibration_bins
                == evidence
            )
        )

        values = (
            calibration[
                "scores"
            ][selected]
        )

        # Safety fallback.
        if len(values) < 100:

            threshold = (
                global_threshold
            )

        else:

            threshold = (
                conformal_quantile(
                    values,
                    ALPHA,
                )
            )

        thresholds[
            evidence
        ] = float(
            threshold
        )

    return thresholds


# ============================================================
# METRICS
# ============================================================

def metrics(
    labels,
    scores,
    predictions,
):

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
        if fp + tn > 0
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
    }


def fpr_by_evidence(
    labels,
    predictions,
    bins,
):

    result = {}

    names = {
        0: "low",
        1: "medium",
        2: "high",
    }

    for b in [
        0,
        1,
        2,
    ]:

        selected = (
            bins == b
        )

        y = labels[
            selected
        ]

        p = predictions[
            selected
        ]

        normal = (
            y == 0
        )

        n_normal = int(
            normal.sum()
        )

        if n_normal == 0:

            fpr = None

        else:

            fp = int(
                (
                    (p == 1)
                    & normal
                ).sum()
            )

            fpr = (
                fp / n_normal
            )

        result[
            names[b]
        ] = {
            "normal_rows":
                n_normal,

            "fpr":
                (
                    float(fpr)
                    if fpr is not None
                    else None
                ),
        }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print(
        "STAGE 3.8 — "
        "EVIDENCE-AWARE CONFORMAL CALIBRATION"
    )
    print("=" * 72)

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

    all_results = []

    for (
        condition_name,
        loss_type,
        loss_rate,
    ) in conditions:

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"CONDITION: "
            f"{condition_name}"
        )

        print(
            "=" * 72
        )

        # ----------------------------------------------------
        # VALIDATION / CALIBRATION
        # ----------------------------------------------------

        print(
            "\n  Collecting calibration data..."
        )

        calibration = collect(
            val_manifest,
            loss_type,
            loss_rate,
        )

        boundaries = (
            fit_evidence_boundaries(
                calibration
            )
        )

        calibration_bins = (
            evidence_bins(
                calibration[
                    "gaps"
                ],
                calibration[
                    "uncertainty"
                ],
                boundaries,
            )
        )

        global_threshold = (
            fit_global_threshold(
                calibration
            )
        )

        evidence_thresholds = (
            fit_evidence_thresholds(
                calibration,
                calibration_bins,
            )
        )

        print(
            f"\n  Global threshold: "
            f"{global_threshold:.6f}"
        )

        print(
            "  Evidence thresholds:"
        )

        print(
            f"    low    : "
            f"{evidence_thresholds[0]:.6f}"
        )

        print(
            f"    medium : "
            f"{evidence_thresholds[1]:.6f}"
        )

        print(
            f"    high   : "
            f"{evidence_thresholds[2]:.6f}"
        )

        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        print(
            "\n  Collecting test data..."
        )

        test = collect(
            test_manifest,
            loss_type,
            loss_rate,
        )

        test_bins = (
            evidence_bins(
                test[
                    "gaps"
                ],
                test[
                    "uncertainty"
                ],
                boundaries,
            )
        )

        labels = (
            test[
                "labels"
            ]
        )

        scores = (
            test[
                "scores"
            ]
        )

        # ----------------------------------------------------
        # GLOBAL CONFORMAL
        # ----------------------------------------------------

        global_predictions = (
            scores
            >= global_threshold
        ).astype(
            np.int64
        )

        global_metrics = (
            metrics(
                labels,
                scores,
                global_predictions,
            )
        )

        global_fpr_bins = (
            fpr_by_evidence(
                labels,
                global_predictions,
                test_bins,
            )
        )

        # ----------------------------------------------------
        # EVIDENCE-CONDITIONED CONFORMAL
        # ----------------------------------------------------

        conditioned_predictions = (
            np.zeros(
                len(scores),
                dtype=np.int64,
            )
        )

        for evidence in [
            0,
            1,
            2,
        ]:

            selected = (
                test_bins
                == evidence
            )

            conditioned_predictions[
                selected
            ] = (
                scores[
                    selected
                ]
                >= evidence_thresholds[
                    evidence
                ]
            ).astype(
                np.int64
            )

        conditioned_metrics = (
            metrics(
                labels,
                scores,
                conditioned_predictions,
            )
        )

        conditioned_fpr_bins = (
            fpr_by_evidence(
                labels,
                conditioned_predictions,
                test_bins,
            )
        )

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "\n  GLOBAL CALIBRATION"
        )

        print(
            f"    F1     : "
            f"{global_metrics['f1']:.4f}"
        )

        print(
            f"    Recall : "
            f"{global_metrics['recall']:.4f}"
        )

        print(
            f"    FPR    : "
            f"{global_metrics['fpr']:.4f}"
        )

        print(
            "\n  EVIDENCE-CONDITIONED"
        )

        print(
            f"    F1     : "
            f"{conditioned_metrics['f1']:.4f}"
        )

        print(
            f"    Recall : "
            f"{conditioned_metrics['recall']:.4f}"
        )

        print(
            f"    FPR    : "
            f"{conditioned_metrics['fpr']:.4f}"
        )

        print(
            "\n  FPR BY EVIDENCE"
        )

        for name in [
            "low",
            "medium",
            "high",
        ]:

            g = (
                global_fpr_bins[
                    name
                ]["fpr"]
            )

            c = (
                conditioned_fpr_bins[
                    name
                ]["fpr"]
            )

            print(
                f"    {name:<7} "
                f"global="
                f"{g if g is not None else 'NA'} "
                f"conditioned="
                f"{c if c is not None else 'NA'}"
            )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        all_results.append({

            "condition":
                condition_name,

            "loss_type":
                loss_type,

            "loss_rate":
                loss_rate,

            "boundaries":
                boundaries,

            "global_threshold":
                global_threshold,

            "evidence_thresholds":
                evidence_thresholds,

            "global_metrics":
                global_metrics,

            "conditioned_metrics":
                conditioned_metrics,

            "global_fpr_by_evidence":
                global_fpr_bins,

            "conditioned_fpr_by_evidence":
                conditioned_fpr_bins,
        })

    # ========================================================
    # SAVE
    # ========================================================

    output_dir = Path(
        "results/evidence_conformal"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "evidence_conformal_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n\n"
        + "=" * 95
    )

    print(
        "FINAL EVIDENCE-CONFORMAL COMPARISON"
    )

    print(
        "=" * 95
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
        "-" * 95
    )

    for r in all_results:

        gm = r[
            "global_metrics"
        ]

        cm = r[
            "conditioned_metrics"
        ]

        print(
            f"{r['condition']:<15}"
            f"{gm['f1']:>12.4f}"
            f"{cm['f1']:>12.4f}"
            f"{gm['fpr']:>14.4f}"
            f"{cm['fpr']:>12.4f}"
            f"{gm['recall']:>14.4f}"
            f"{cm['recall']:>12.4f}"
        )

    print(
        "\nSaved to:"
    )

    print(
        output_file.resolve()
    )

    print(
        "\nSTAGE 3.8 COMPLETE"
    )


if __name__ == "__main__":
    main()
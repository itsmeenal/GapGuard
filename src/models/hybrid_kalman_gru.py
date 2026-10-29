import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data.experiment_loader import load_manifest, load_trajectory


# ============================================================
# CONFIG
# ============================================================

SEED = 42

WINDOW_SIZE = 8
HIDDEN_SIZE = 24

EPOCHS = 5
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

MAX_TRAIN_WINDOWS = 150_000

FIXED_DT = 5.0
MIN_DT = 0.1
MAX_DT = 120.0

# [latitude, longitude, altitude, speed]
MEASUREMENT_COLUMNS = [
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
]

MEASUREMENT_SCALE = np.array(
    [1.0, 1.0, 100.0, 50.0],
    dtype=np.float32,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA HELPERS
# ============================================================

def get_dataframe(loader_output):
    """
    Extract DataFrame from experiment_loader output.
    """

    if isinstance(loader_output, pd.DataFrame):
        return loader_output

    if isinstance(loader_output, (tuple, list)):

        for item in loader_output:

            if isinstance(item, pd.DataFrame):
                return item

    raise TypeError(
        f"Could not extract DataFrame from "
        f"{type(loader_output)}"
    )


def get_measurements(df):
    """
    Extract the four detector measurements
    and put them on comparable scales.
    """

    values = df[
        MEASUREMENT_COLUMNS
    ].to_numpy(dtype=np.float32)

    return values / MEASUREMENT_SCALE


# ============================================================
# SIMPLE DELTA-T KALMAN
# ============================================================

class DeltaTKalman:

    def __init__(self):

        self.state = None

        self.P = np.eye(
            4,
            dtype=np.float64
        )

        self.Q_base = np.diag([
            0.01,
            0.01,
            0.10,
            0.10,
        ])

        self.R = np.diag([
            0.01,
            0.01,
            0.10,
            0.10,
        ])

    def reset(self):

        self.state = None

        self.P = np.eye(
            4,
            dtype=np.float64
        )

    def update(self, measurement, dt):

        z = np.asarray(
            measurement,
            dtype=np.float64
        )

        if self.state is None:

            self.state = z.copy()

            return (
                z.copy(),
                np.zeros(4),
                0.0,
            )

        effective_dt = float(
            np.clip(
                dt,
                MIN_DT,
                MAX_DT,
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        F = np.eye(
            4,
            dtype=np.float64
        )

        Q = (
            self.Q_base
            * (
                effective_dt
                / FIXED_DT
            )
        )

        predicted_state = (
            F @ self.state
        )

        predicted_P = (
            F @ self.P @ F.T
            + Q
        )

        # ----------------------------------------------------
        # Innovation
        # ----------------------------------------------------

        innovation = (
            z - predicted_state
        )

        S = (
            predicted_P
            + self.R
        )

        try:

            S_inv = np.linalg.inv(S)

        except np.linalg.LinAlgError:

            S_inv = np.linalg.pinv(S)

        # Mahalanobis severity
        severity = float(
            innovation.T
            @ S_inv
            @ innovation
        )

        # ----------------------------------------------------
        # Kalman update
        # ----------------------------------------------------

        K = (
            predicted_P
            @ S_inv
        )

        self.state = (
            predicted_state
            + K @ innovation
        )

        I = np.eye(4)

        self.P = (
            (I - K)
            @ predicted_P
        )

        return (
            predicted_state,
            innovation,
            severity,
        )


# ============================================================
# BUILD KALMAN RESIDUAL SEQUENCES
# ============================================================

def get_residual_sequence(df):

    values = get_measurements(df)

    times = df[
        "t_elapsed_sec"
    ].to_numpy(
        dtype=np.float64
    )

    kalman = DeltaTKalman()

    residuals = []
    severities = []

    for i in range(len(values)):

        if i == 0:

            dt = FIXED_DT

        else:

            dt = (
                times[i]
                - times[i - 1]
            )

        prediction, residual, severity = (
            kalman.update(
                values[i],
                dt,
            )
        )

        residuals.append(residual)
        severities.append(severity)

    return (
        np.asarray(
            residuals,
            dtype=np.float32,
        ),
        np.asarray(
            severities,
            dtype=np.float32,
        ),
    )


# ============================================================
# GRU
# ============================================================

class ResidualGRU(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=24,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        self.fc = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last = output[:, -1, :]

        return self.fc(last)


# ============================================================
# TRAINING WINDOWS
# ============================================================

def collect_training_windows(
    train_manifest,
    max_windows=MAX_TRAIN_WINDOWS,
):

    X = []
    Y = []

    total = 0

    print(
        "\nBuilding NORMAL Kalman-residual windows..."
    )

    for idx, (_, row) in enumerate(
        train_manifest.iterrows()
    ):

        loaded = load_trajectory(row)

        df = get_dataframe(loaded)

        # ----------------------------------------------------
        # IMPORTANT:
        # only completely normal trajectories/observations
        # are used for training.
        # ----------------------------------------------------

        normal_df = df[
            df["label"] == 0
        ].copy()

        if len(normal_df) <= WINDOW_SIZE:
            continue

        residuals, _ = (
            get_residual_sequence(
                normal_df
            )
        )

        for i in range(
            WINDOW_SIZE,
            len(residuals),
        ):

            X.append(
                residuals[
                    i - WINDOW_SIZE:i
                ]
            )

            Y.append(
                residuals[i]
            )

            total += 1

            if total >= max_windows:
                break

        if total >= max_windows:
            break

        if (idx + 1) % 500 == 0:

            print(
                f"  processed "
                f"{idx + 1}/"
                f"{len(train_manifest)}"
            )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    Y = np.asarray(
        Y,
        dtype=np.float32,
    )

    return X, Y


# ============================================================
# TRAIN GRU
# ============================================================

def train_model(X, Y):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Training device: {device}"
    )

    model = ResidualGRU(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    dataset = (
        torch.utils.data.TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y),
        )
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = criterion(
                prediction,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_x)
            )

        mean_loss = (
            total_loss
            / len(dataset)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {mean_loss:.6f}"
        )

    return model, device


# ============================================================
# HYBRID SCORING
# ============================================================

def score_trajectory(
    model,
    device,
    df,
):

    residuals, kalman_scores = (
        get_residual_sequence(df)
    )

    labels = (
        df["label"]
        .to_numpy()
        .astype(np.int64)
    )

    labels = (
        labels > 0
    ).astype(np.int64)

    hybrid_scores = np.zeros(
        len(df),
        dtype=np.float64,
    )

    if len(df) <= WINDOW_SIZE:

        return (
            hybrid_scores,
            labels,
            kalman_scores,
        )

    model.eval()

    with torch.no_grad():

        windows = []

        indices = []

        for i in range(
            WINDOW_SIZE,
            len(residuals),
        ):

            windows.append(
                residuals[
                    i - WINDOW_SIZE:i
                ]
            )

            indices.append(i)

        windows = np.asarray(
            windows,
            dtype=np.float32,
        )

        tensor_x = (
            torch.from_numpy(
                windows
            ).to(device)
        )

        predicted_residual = (
            model(tensor_x)
            .cpu()
            .numpy()
        )

        actual_residual = (
            residuals[
                WINDOW_SIZE:
            ]
        )

        # ----------------------------------------------------
        # CORRECTED RESIDUAL
        # ----------------------------------------------------

        corrected = (
            actual_residual
            - predicted_residual
        )

        # Squared normalized residual
        corrected_error = np.mean(
            corrected ** 2,
            axis=1,
        )

        hybrid_scores[
            WINDOW_SIZE:
        ] = corrected_error

    return (
        hybrid_scores,
        labels,
        kalman_scores,
    )


# ============================================================
# COLLECT SCORES
# ============================================================

def collect_scores(
    manifest,
    model,
    device,
    name,
):

    all_scores = []
    all_labels = []
    all_kalman = []

    print(
        f"Scoring {name}: "
        f"{len(manifest):,} trajectories..."
    )

    for idx, (_, row) in enumerate(
        manifest.iterrows()
    ):

        loaded = load_trajectory(row)

        df = get_dataframe(loaded)

        scores, labels, kalman_scores = (
            score_trajectory(
                model,
                device,
                df,
            )
        )

        all_scores.append(scores)
        all_labels.append(labels)
        all_kalman.append(kalman_scores)

        if (idx + 1) % 500 == 0:

            print(
                f"  processed "
                f"{idx + 1:,}/"
                f"{len(manifest):,}"
            )

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_kalman),
    )


# ============================================================
# THRESHOLD
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
    ).astype(np.int64)

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
    )

    return {
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "roc_auc": float(
            roc_auc
        ),
        "pr_auc": float(
            pr_auc
        ),
        "fpr": float(
            fpr
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_alerts": int(
            predictions.sum()
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "STAGE 3.6 — Δt-KALMAN + GRU RESIDUAL"
    )
    print("=" * 60)

    manifest = load_manifest()

    train_manifest = (
        manifest[
            manifest["split"] == "train"
        ]
        .reset_index(drop=True)
    )

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

    print(
        f"\nTrain trajectories : "
        f"{len(train_manifest):,}"
    )

    print(
        f"Val trajectories   : "
        f"{len(val_manifest):,}"
    )

    print(
        f"Test trajectories  : "
        f"{len(test_manifest):,}"
    )

    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    X_train, Y_train = (
        collect_training_windows(
            train_manifest
        )
    )

    print(
        f"\nTraining windows : "
        f"{len(X_train):,}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining residual GRU..."
    )

    model, device = train_model(
        X_train,
        Y_train,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\nEvaluating validation data..."
    )

    val_scores, val_labels, _ = (
        collect_scores(
            val_manifest,
            model,
            device,
            "validation",
        )
    )

    threshold = select_threshold(
        val_scores,
        val_labels,
    )

    print(
        f"\nHybrid threshold : "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating clean test data..."
    )

    test_scores, test_labels, test_kalman = (
        collect_scores(
            test_manifest,
            model,
            device,
            "test",
        )
    )

    hybrid_results = evaluate(
        test_scores,
        test_labels,
        threshold,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "Δt-KALMAN + GRU RESIDUAL — CLEAN TEST"
    )
    print("=" * 60)

    print(
        f"Test rows       : "
        f"{len(test_labels):,}"
    )

    print(
        f"Attack rows     : "
        f"{int(test_labels.sum()):,}"
    )

    print(
        f"Predicted alerts: "
        f"{hybrid_results['predicted_alerts']:,}"
    )

    print(
        f"\nPrecision : "
        f"{hybrid_results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{hybrid_results['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{hybrid_results['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{hybrid_results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{hybrid_results['pr_auc']:.4f}"
    )

    print(
        f"FPR       : "
        f"{hybrid_results['fpr']:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        f"TN={hybrid_results['true_negative']}, "
        f"FP={hybrid_results['false_positive']}, "
        f"FN={hybrid_results['false_negative']}, "
        f"TP={hybrid_results['true_positive']}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_dir = Path(
        "results/hybrid_kalman_gru"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "clean_test_result.json"
    )

    output = {
        "stage": "3.6",
        "model": "delta_t_kalman_gru_residual",
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "training_windows": int(
            len(X_train)
        ),
        "threshold": threshold,
        **hybrid_results,
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

    print(
        "\nResult saved to:"
    )

    print(
        output_file.resolve()
    )

    print(
        "\nSTAGE 3.6 COMPLETE"
    )


if __name__ == "__main__":
    main()
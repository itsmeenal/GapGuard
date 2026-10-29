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

from src.data.experiment_loader import (
    load_manifest,
    load_trajectory,
    FEATURE_COLUMNS,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

WINDOW_SIZE = 10
HIDDEN_SIZE = 32
NUM_LAYERS = 1
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

MAX_TRAIN_ROWS = 200_000
MAX_VAL_ROWS = 50_000

# We use the same four measurement variables as Stage 3.4.
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
# HELPERS
# ============================================================

def get_dataframe(loader_output):
    """
    experiment_loader may return either a DataFrame directly
    or a tuple/list containing the DataFrame.
    """
    if isinstance(loader_output, pd.DataFrame):
        return loader_output

    if isinstance(loader_output, (tuple, list)):
        for item in loader_output:
            if isinstance(item, pd.DataFrame):
                return item

    raise TypeError(
        f"Could not find DataFrame in loader output: "
        f"{type(loader_output)}"
    )


def get_measurements(df):
    """
    Extract and scale the four detector measurements.
    """
    values = df[MEASUREMENT_COLUMNS].to_numpy(
        dtype=np.float32
    )

    values = values / MEASUREMENT_SCALE

    return values


def binary_labels(df):
    """
    Dataset labels:
        0 = normal
        1/2/3 = injected anomaly

    Convert to binary for evaluation.
    """
    return (df["label"].to_numpy() > 0).astype(np.int64)


# ============================================================
# GRU MODEL
# ============================================================

class ResidualGRU(nn.Module):
    """
    Small GRU that predicts the next normalized observation.

    Input:
        previous WINDOW_SIZE observations

    Output:
        predicted next observation
    """

    def __init__(
        self,
        input_size=4,
        hidden_size=32,
        num_layers=1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, x):
        output, _ = self.gru(x)

        last_hidden = output[:, -1, :]

        prediction = self.output(last_hidden)

        return prediction


# ============================================================
# BUILD TRAINING WINDOWS
# ============================================================

def collect_training_windows(
    manifest,
    max_rows=200_000,
):
    """
    Build next-step prediction windows using NORMAL rows only.

    Important:
    Attack rows are never used for training.
    """

    X = []
    Y = []

    total_rows = 0

    for idx, (_, row) in enumerate(manifest.iterrows()):

        loaded = load_trajectory(row)
        df = get_dataframe(loaded)

        # Keep only normal observations.
        normal_df = df[df["label"] == 0]

        if len(normal_df) <= WINDOW_SIZE:
            continue

        values = get_measurements(normal_df)

        for i in range(WINDOW_SIZE, len(values)):

            X.append(
                values[i - WINDOW_SIZE:i]
            )

            Y.append(
                values[i]
            )

            total_rows += 1

            if total_rows >= max_rows:
                break

        if total_rows >= max_rows:
            break

        if (idx + 1) % 500 == 0:
            print(
                f"  training trajectories processed "
                f"{idx + 1}/{len(manifest)}"
            )

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    return X, Y


# ============================================================
# TRAIN
# ============================================================

def train_model(X, Y):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Training device: {device}")

    model = ResidualGRU(
        input_size=4,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(Y),
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()

    for epoch in range(EPOCHS):

        running_loss = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            prediction = model(batch_x)

            loss = criterion(
                prediction,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * len(batch_x)
            )

        epoch_loss = (
            running_loss / len(dataset)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {epoch_loss:.6f}"
        )

    return model, device


# ============================================================
# SCORE ONE TRAJECTORY
# ============================================================

def score_trajectory(
    model,
    device,
    df,
):
    """
    Generate GRU prediction-error scores.

    First WINDOW_SIZE observations receive score 0
    because there is not enough history yet.
    """

    values = get_measurements(df)

    labels = binary_labels(df)

    scores = np.zeros(
        len(values),
        dtype=np.float64,
    )

    if len(values) <= WINDOW_SIZE:
        return scores, labels

    model.eval()

    with torch.no_grad():

        windows = []

        indices = []

        for i in range(
            WINDOW_SIZE,
            len(values),
        ):

            windows.append(
                values[
                    i - WINDOW_SIZE:i
                ]
            )

            indices.append(i)

        windows = np.asarray(
            windows,
            dtype=np.float32,
        )

        tensor_x = torch.from_numpy(
            windows
        ).to(device)

        predictions = model(
            tensor_x
        ).cpu().numpy()

        actual = values[
            WINDOW_SIZE:
        ]

        errors = np.mean(
            (predictions - actual) ** 2,
            axis=1,
        )

        scores[
            WINDOW_SIZE:
        ] = errors

    return scores, labels


# ============================================================
# COLLECT SCORES
# ============================================================

def collect_scores(
    manifest,
    model,
    device,
    max_rows=None,
    name="data",
):

    all_scores = []
    all_labels = []

    processed_rows = 0

    print(
        f"Scoring {name}: "
        f"{len(manifest):,} trajectories..."
    )

    for idx, (_, row) in enumerate(
        manifest.iterrows()
    ):

        loaded = load_trajectory(row)
        df = get_dataframe(loaded)

        scores, labels = score_trajectory(
            model,
            device,
            df,
        )

        all_scores.append(scores)
        all_labels.append(labels)

        processed_rows += len(df)

        if (
            idx + 1
        ) % 500 == 0:

            print(
                f"  processed "
                f"{idx + 1:,}/"
                f"{len(manifest):,}"
            )

        if (
            max_rows is not None
            and processed_rows >= max_rows
        ):
            break

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
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

    threshold = np.percentile(
        normal_scores,
        99,
    )

    return float(threshold)


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

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
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
        "STAGE 3.5 — NEURAL RESIDUAL GRU"
    )
    print("=" * 60)

    manifest = load_manifest()

    train_manifest = manifest[
        manifest["split"] == "train"
    ].reset_index(drop=True)

    val_manifest = manifest[
        manifest["split"] == "val"
    ].reset_index(drop=True)

    test_manifest = manifest[
        manifest["split"] == "test"
    ].reset_index(drop=True)

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

    print(
        "\nBuilding NORMAL training windows..."
    )

    X_train, Y_train = (
        collect_training_windows(
            train_manifest,
            max_rows=MAX_TRAIN_ROWS,
        )
    )

    print(
        f"Training windows : "
        f"{len(X_train):,}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining GRU...")

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

    val_scores, val_labels = collect_scores(
        val_manifest,
        model,
        device,
        max_rows=MAX_VAL_ROWS,
        name="validation",
    )

    threshold = select_threshold(
        val_scores,
        val_labels,
    )

    print(
        f"\nGRU threshold : "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating clean test data..."
    )

    test_scores, test_labels = (
        collect_scores(
            test_manifest,
            model,
            device,
            name="test",
        )
    )

    results = evaluate(
        test_scores,
        test_labels,
        threshold,
    )

    print("\n" + "=" * 60)
    print(
        "NEURAL RESIDUAL GRU — CLEAN TEST"
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
        f"{results['predicted_alerts']:,}"
    )

    print(
        f"\nPrecision : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{results['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{results['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC    : "
        f"{results['pr_auc']:.4f}"
    )

    print(
        f"FPR       : "
        f"{results['fpr']:.4f}"
    )

    print("\nConfusion matrix:")

    print(
        f"TN={results['true_negative']}, "
        f"FP={results['false_positive']}, "
        f"FN={results['false_negative']}, "
        f"TP={results['true_positive']}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_dir = Path(
        "results/neural_residual"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir /
        "clean_test_result.json"
    )

    output = {
        "stage": "3.5",
        "model": "GRU_next_step_prediction",
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "epochs": EPOCHS,
        "training_rows": len(X_train),
        "threshold": threshold,
        **results,
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
        "\nSTAGE 3.5 COMPLETE"
    )


if __name__ == "__main__":
    main()
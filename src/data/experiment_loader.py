from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAJECTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectories"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_manifest.csv"
)


# ============================================================
# FEATURES
# ============================================================

# These are the ONLY variables that anomaly detectors
# are allowed to see.
#
# Do NOT add:
#   label
#   delta_altitude
#   delta_speed
#   delta_heading
#   x_wrt0
#   y_wrt0
#   route_progress
#   t_norm
#
# Those variables are either labels or derived variables
# that we deliberately excluded to avoid leakage.

FEATURE_COLUMNS = [
    "t_elapsed_sec",
    "dt",
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "heading",
]


# Label is kept separately for evaluation.
# It is NEVER passed to a detector.
LABEL_COLUMN = "label"


# ============================================================
# LOAD CLEAN MANIFEST
# ============================================================

def load_manifest():
    """
    Load the cleaned manifest.

    The manifest defines:
        route
        flight_id
        split
        csv_path
        file_label
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Clean manifest not found:\n{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required_columns = {
        "route",
        "flight_id",
        "split",
        "csv_path",
        "file_label",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["flight_id"] = (
        manifest["flight_id"]
        .astype(str)
    )

    manifest["file_label"] = (
        pd.to_numeric(
            manifest["file_label"],
            errors="raise"
        )
        .astype(int)
    )

    manifest["split"] = (
        manifest["split"]
        .astype(str)
        .str.lower()
    )

    return manifest


# ============================================================
# FIND PROCESSED TRAJECTORY
# ============================================================

def trajectory_path(row):
    """
    Convert a manifest row into the corresponding
    processed trajectory path.

    IMPORTANT:
    The preprocessing stage created files using:

        <route>_<flight_id>_v<file_label>.csv

    Examples:

        ceb_3e2f2d34_v0.csv
        ceb_3e2f2d34_v1.csv
        ceb_3e2f2d34_v2.csv
        ceb_3e2f2d34_v3.csv
    """

    path = TRAJECTORY_DIR / (
        f"{row['route']}_"
        f"{row['flight_id']}_"
        f"v{row['file_label']}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Processed trajectory not found:\n"
            f"{path}"
        )

    return path


# ============================================================
# LOAD ONE TRAJECTORY
# ============================================================

def load_trajectory(row):
    """
    Load one processed trajectory.

    Returns
    -------
    features : numpy array
        Detector input features.

    labels : numpy array
        Ground-truth labels used only for evaluation.

    dataframe : pandas DataFrame
        Selected trajectory data for bookkeeping.
    """

    path = trajectory_path(row)

    df = pd.read_csv(path)

    required_columns = set(
        FEATURE_COLUMNS + [LABEL_COLUMN]
    )

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Trajectory {path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    # Keep ONLY detector features + label.
    df = df[
        FEATURE_COLUMNS + [LABEL_COLUMN]
    ].copy()

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    df[LABEL_COLUMN] = pd.to_numeric(
        df[LABEL_COLUMN],
        errors="raise"
    ).astype(int)

    # --------------------------------------------------------
    # NaN check
    # --------------------------------------------------------

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            f"NaN detected in trajectory: {path.name}"
        )

    # --------------------------------------------------------
    # Infinite-value check
    # --------------------------------------------------------

    feature_values = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Non-finite value detected in trajectory: "
            f"{path.name}"
        )

    # --------------------------------------------------------
    # Time validation
    # --------------------------------------------------------

    time_values = df[
        "t_elapsed_sec"
    ].to_numpy(dtype=float)

    if len(time_values) > 1:

        time_difference = np.diff(
            time_values
        )

        if np.any(time_difference <= 0):
            raise ValueError(
                "Non-increasing time detected in "
                f"trajectory: {path.name}"
            )

    # --------------------------------------------------------
    # dt validation
    # --------------------------------------------------------

    dt_values = df[
        "dt"
    ].to_numpy(dtype=float)

    if len(df) > 1:

        expected_dt = np.diff(
            time_values
        )

        if not np.allclose(
            dt_values[1:],
            expected_dt,
            atol=1e-9
        ):
            raise ValueError(
                f"dt mismatch in trajectory: {path.name}"
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    features = df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    labels = df[
        LABEL_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    return features, labels, df


# ============================================================
# GET SPLIT MANIFEST
# ============================================================

def get_split_manifest(split):
    """
    Return manifest rows belonging to one split.

    Valid values:
        train
        val
        test
    """

    split = str(split).lower()

    if split not in {
        "train",
        "val",
        "test",
    }:
        raise ValueError(
            "split must be one of: "
            "train, val, test"
        )

    manifest = load_manifest()

    result = manifest[
        manifest["split"] == split
    ].copy()

    if result.empty:
        raise ValueError(
            f"No trajectories found for split='{split}'"
        )

    return result.reset_index(
        drop=True
    )


# ============================================================
# LOAD ALL TRAJECTORIES IN A SPLIT
# ============================================================

def load_split(
    split,
    normal_only=False
):
    """
    Load all trajectories belonging to a split.

    Parameters
    ----------
    split : str
        train / val / test

    normal_only : bool
        If True, only rows with label == 0
        are retained.

    IMPORTANT:
        Labels are used ONLY to construct the training
        subset. Labels are never included in features.
    """

    manifest = get_split_manifest(
        split
    )

    trajectories = []

    for _, row in manifest.iterrows():

        features, labels, df = (
            load_trajectory(row)
        )

        if normal_only:

            normal_mask = (
                labels == 0
            )

            features = (
                features[normal_mask]
            )

            labels = (
                labels[normal_mask]
            )

            df = (
                df.loc[
                    normal_mask
                ]
                .reset_index(drop=True)
            )

        trajectories.append(
            {
                "route": row["route"],
                "flight_id": row["flight_id"],
                "file_label": int(
                    row["file_label"]
                ),
                "split": row["split"],
                "features": features,
                "labels": labels,
                "dataframe": df,
            }
        )

    return trajectories


# ============================================================
# DETERMINISTIC SEED
# ============================================================

def make_seed(
    identifier,
    loss_type,
    loss_rate
):
    """
    Create a deterministic random seed.

    The same:
        trajectory
        loss type
        loss rate

    will always produce the same loss mask.
    """

    text = (
        f"{identifier}|"
        f"{loss_type}|"
        f"{float(loss_rate):.4f}"
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16
    )


# ============================================================
# MAKE PACKET-LOSS MASK
# ============================================================

def make_loss_mask(
    n_rows,
    loss_rate,
    loss_type="random",
    identifier="trajectory",
):
    """
    Generate a deterministic packet-loss mask.

    True
        observation received

    False
        observation lost

    The first observation is ALWAYS retained.
    """

    if n_rows <= 0:
        return np.array(
            [],
            dtype=bool
        )

    if not 0 <= loss_rate < 1:
        raise ValueError(
            "loss_rate must satisfy "
            "0 <= loss_rate < 1"
        )

    if loss_type not in {
        "random",
        "burst",
    }:
        raise ValueError(
            "loss_type must be "
            "'random' or 'burst'"
        )

    seed = make_seed(
        identifier,
        loss_type,
        loss_rate
    )

    rng = np.random.default_rng(
        seed
    )

    # Start with everything received.
    mask = np.ones(
        n_rows,
        dtype=bool
    )

    if (
        n_rows == 1
        or loss_rate == 0
    ):
        return mask

    # Number of rows that should be lost.
    target_lost = int(
        round(
            n_rows * loss_rate
        )
    )

    # First observation must remain.
    target_lost = min(
        target_lost,
        n_rows - 1
    )

    if target_lost <= 0:
        return mask

    # --------------------------------------------------------
    # RANDOM LOSS
    # --------------------------------------------------------

    if loss_type == "random":

        candidates = np.arange(
            1,
            n_rows
        )

        selected = rng.choice(
            candidates,
            size=target_lost,
            replace=False
        )

        mask[selected] = False

    # --------------------------------------------------------
    # BURST LOSS
    # --------------------------------------------------------

    else:

        lost = 0

        while lost < target_lost:

            available = np.flatnonzero(
                mask
            )

            # Never delete first observation.
            available = available[
                available > 0
            ]

            if len(available) == 0:
                break

            start = int(
                rng.choice(
                    available
                )
            )

            remaining = (
                target_lost - lost
            )

            max_burst = min(
                8,
                remaining
            )

            min_burst = min(
                2,
                max_burst
            )

            if (
                min_burst
                == max_burst
            ):

                burst_length = (
                    min_burst
                )

            else:

                burst_length = int(
                    rng.integers(
                        min_burst,
                        max_burst + 1
                    )
                )

            end = min(
                start + burst_length,
                n_rows
            )

            indices = np.arange(
                start,
                end
            )

            # Keep only observations
            # that have not already been lost.
            indices = indices[
                mask[indices]
            ]

            if len(indices) == 0:
                continue

            mask[indices] = False

            lost += len(indices)

    return mask


# ============================================================
# APPLY LOSS
# ============================================================

def apply_loss(
    features,
    labels,
    loss_rate,
    loss_type,
    identifier,
):
    """
    Apply deterministic packet loss in memory.

    Returns
    -------
    received_features
    received_labels
    received_mask
    """

    mask = make_loss_mask(
        n_rows=len(features),
        loss_rate=loss_rate,
        loss_type=loss_type,
        identifier=identifier,
    )

    received_features = (
        features[mask]
    )

    received_labels = (
        labels[mask]
    )

    return (
        received_features,
        received_labels,
        mask,
    )


# ============================================================
# LOAD EXPERIMENT SPLIT
# ============================================================

def load_experiment_split(
    split,
    loss_rate=0.0,
    loss_type="none",
    normal_only=False,
):
    """
    Load a complete experiment split.

    Parameters
    ----------
    split :
        train / val / test

    loss_rate :
        0.0 to <1.0

    loss_type :
        none / random / burst

    normal_only :
        retain only normal observations

    Packet loss is generated IN MEMORY.
    """

    trajectories = load_split(
        split,
        normal_only=normal_only
    )

    # --------------------------------------------------------
    # No packet loss
    # --------------------------------------------------------

    if (
        loss_type == "none"
        or loss_rate == 0
    ):

        for trajectory in trajectories:

            trajectory[
                "loss_rate"
            ] = 0.0

            trajectory[
                "loss_type"
            ] = "none"

            trajectory[
                "received_mask"
            ] = np.ones(
                len(
                    trajectory[
                        "features"
                    ]
                ),
                dtype=bool
            )

        return trajectories

    # --------------------------------------------------------
    # Validate loss type
    # --------------------------------------------------------

    if loss_type not in {
        "random",
        "burst",
    }:
        raise ValueError(
            "loss_type must be "
            "none, random, or burst"
        )

    processed = []

    for trajectory in trajectories:

        identifier = (
            f"{trajectory['route']}_"
            f"{trajectory['flight_id']}_"
            f"{trajectory['file_label']}"
        )

        (
            received_features,
            received_labels,
            mask,
        ) = apply_loss(
            trajectory[
                "features"
            ],
            trajectory[
                "labels"
            ],
            loss_rate,
            loss_type,
            identifier,
        )

        # Preserve original data for evaluation.
        trajectory[
            "original_features"
        ] = trajectory[
            "features"
        ]

        trajectory[
            "original_labels"
        ] = trajectory[
            "labels"
        ]

        # Replace active data with received observations.
        trajectory[
            "features"
        ] = received_features

        trajectory[
            "labels"
        ] = received_labels

        trajectory[
            "received_mask"
        ] = mask

        trajectory[
            "loss_rate"
        ] = float(
            loss_rate
        )

        trajectory[
            "loss_type"
        ] = loss_type

        processed.append(
            trajectory
        )

    return processed


# ============================================================
# SUMMARY
# ============================================================

def summarize_trajectories(
    trajectories
):
    """
    Print a compact summary of trajectories.
    """

    if not trajectories:

        print(
            "No trajectories loaded."
        )

        return

    total_rows = sum(
        len(
            item["features"]
        )
        for item in trajectories
    )

    total_attack_rows = sum(
        np.sum(
            item["labels"] != 0
        )
        for item in trajectories
    )

    number_of_flights = len(
        trajectories
    )

    print(
        "=" * 60
    )

    print(
        "EXPERIMENT DATA SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"Trajectories : "
        f"{number_of_flights:,}"
    )

    print(
        f"Rows         : "
        f"{total_rows:,}"
    )

    print(
        f"Attack rows  : "
        f"{total_attack_rows:,}"
    )

    print(
        f"Average rows/trajectory : "
        f"{total_rows / number_of_flights:.2f}"
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "STAGE 3.1 — EXPERIMENT LOADER TEST"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    print(
        f"Manifest rows : "
        f"{len(manifest):,}"
    )

    print(
        f"Train : "
        f"{(manifest['split'] == 'train').sum():,}"
    )

    print(
        f"Val   : "
        f"{(manifest['split'] == 'val').sum():,}"
    )

    print(
        f"Test  : "
        f"{(manifest['split'] == 'test').sum():,}"
    )

    # --------------------------------------------------------
    # Load one test trajectory
    # --------------------------------------------------------

    print(
        "\nLoading 1 test trajectory..."
    )

    test_manifest = (
        get_split_manifest("test")
    )

    row = test_manifest.iloc[0]

    features, labels, df = (
        load_trajectory(row)
    )

    print(
        f"Trajectory : "
        f"{row['route']}_"
        f"{row['flight_id']}_"
        f"v{row['file_label']}"
    )

    print(
        f"Rows       : "
        f"{len(features):,}"
    )

    print(
        f"Features   : "
        f"{features.shape}"
    )

    print(
        f"Labels     : "
        f"{labels.shape}"
    )

    print(
        "Feature columns:"
    )

    for column in FEATURE_COLUMNS:
        print(
            f"  - {column}"
        )

    print(
        f"Unique labels: "
        f"{np.unique(labels).tolist()}"
    )

    # --------------------------------------------------------
    # Random loss test
    # --------------------------------------------------------

    print(
        "\nTesting deterministic random loss..."
    )

    f1, l1, m1 = apply_loss(
        features,
        labels,
        loss_rate=0.30,
        loss_type="random",
        identifier="test_trajectory",
    )

    f2, l2, m2 = apply_loss(
        features,
        labels,
        loss_rate=0.30,
        loss_type="random",
        identifier="test_trajectory",
    )

    print(
        f"Original rows : "
        f"{len(features)}"
    )

    print(
        f"Received rows : "
        f"{len(f1)}"
    )

    print(
        f"Lost rows     : "
        f"{len(features) - len(f1)}"
    )

    print(
        f"Deterministic : "
        f"{np.array_equal(m1, m2)}"
    )

    # --------------------------------------------------------
    # Burst loss test
    # --------------------------------------------------------

    print(
        "\nTesting deterministic burst loss..."
    )

    fb, lb, mb = apply_loss(
        features,
        labels,
        loss_rate=0.30,
        loss_type="burst",
        identifier="test_trajectory",
    )

    print(
        f"Original rows : "
        f"{len(features)}"
    )

    print(
        f"Received rows : "
        f"{len(fb)}"
    )

    print(
        f"Lost rows     : "
        f"{len(features) - len(fb)}"
    )

    # --------------------------------------------------------
    # Burst determinism test
    # --------------------------------------------------------

    fb2, lb2, mb2 = apply_loss(
        features,
        labels,
        loss_rate=0.30,
        loss_type="burst",
        identifier="test_trajectory",
    )

    print(
        f"Burst deterministic : "
        f"{np.array_equal(mb, mb2)}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "STAGE 3.1 LOADER TEST COMPLETE"
    )

    print(
        "=" * 60
    )
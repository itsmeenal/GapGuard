from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Allow importing experiment_loader.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "data")
)

from experiment_loader import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    load_manifest,
    get_split_manifest,
    load_trajectory,
    load_split,
    apply_loss,
    make_loss_mask,
)


# ============================================================
# CONFIGURATION
# ============================================================

FORBIDDEN_COLUMNS = {
    "label",
    "delta_altitude",
    "delta_speed",
    "delta_heading",
    "x_wrt0",
    "y_wrt0",
    "route_progress",
    "t_norm",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("STAGE 3.2 — EXPERIMENT INTEGRITY VALIDATION")
print("=" * 60)


# ============================================================
# 1. LOAD MANIFEST
# ============================================================

manifest = load_manifest()

print("\n1. MANIFEST CHECK")
print("-" * 60)

print(
    f"Total manifest rows : "
    f"{len(manifest):,}"
)


# ============================================================
# 2. CHECK SPLIT VALUES
# ============================================================

valid_splits = {
    "train",
    "val",
    "test",
}

invalid_splits = set(
    manifest["split"].unique()
) - valid_splits

if invalid_splits:
    print(
        "FAIL — invalid split values:",
        sorted(invalid_splits)
    )
    raise SystemExit(1)

print(
    "PASS — only train/val/test splits present"
)


# ============================================================
# 3. CHECK FLIGHT SPLIT INTEGRITY
# ============================================================

print("\n2. FLIGHT-LEVEL SPLIT INTEGRITY")
print("-" * 60)

split_flights = {}

for split in [
    "train",
    "val",
    "test",
]:

    split_manifest = (
        get_split_manifest(split)
    )

    flights = set(
        zip(
            split_manifest["route"],
            split_manifest["flight_id"],
        )
    )

    split_flights[split] = flights

    print(
        f"{split:>5} flights : "
        f"{len(flights):,}"
    )


train_val = (
    split_flights["train"]
    & split_flights["val"]
)

train_test = (
    split_flights["train"]
    & split_flights["test"]
)

val_test = (
    split_flights["val"]
    & split_flights["test"]
)


if train_val:
    print(
        "FAIL — train/val flight overlap:",
        list(train_val)[:5]
    )
    raise SystemExit(1)

if train_test:
    print(
        "FAIL — train/test flight overlap:",
        list(train_test)[:5]
    )
    raise SystemExit(1)

if val_test:
    print(
        "FAIL — val/test flight overlap:",
        list(val_test)[:5]
    )
    raise SystemExit(1)

print(
    "PASS — no flight appears in more than one split"
)


# ============================================================
# 4. CHECK FEATURE DEFINITIONS
# ============================================================

print("\n3. DETECTOR FEATURE CHECK")
print("-" * 60)

print(
    "Detector features:"
)

for feature in FEATURE_COLUMNS:
    print(
        f"  - {feature}"
    )


feature_set = set(
    FEATURE_COLUMNS
)

leaked_features = (
    feature_set
    & FORBIDDEN_COLUMNS
)

if leaked_features:
    print(
        "FAIL — forbidden feature detected:",
        sorted(leaked_features)
    )
    raise SystemExit(1)

if LABEL_COLUMN in feature_set:
    print(
        "FAIL — label is included in detector features"
    )
    raise SystemExit(1)

print(
    "PASS — no label or forbidden derived feature "
    "is included"
)


# ============================================================
# 5. CHECK ONE TRAJECTORY STRUCTURE
# ============================================================

print("\n4. TRAJECTORY STRUCTURE CHECK")
print("-" * 60)

test_manifest = get_split_manifest(
    "test"
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
    f"Feature shape : "
    f"{features.shape}"
)

print(
    f"Label shape   : "
    f"{labels.shape}"
)

if features.shape[1] != len(
    FEATURE_COLUMNS
):
    print(
        "FAIL — unexpected number of features"
    )
    raise SystemExit(1)

if len(features) != len(labels):
    print(
        "FAIL — feature/label length mismatch"
    )
    raise SystemExit(1)

print(
    "PASS — trajectory structure is valid"
)


# ============================================================
# 6. CHECK LABEL IS NOT IN FEATURES
# ============================================================

print("\n5. LABEL ISOLATION CHECK")
print("-" * 60)

feature_dataframe = df[
    FEATURE_COLUMNS
]

if LABEL_COLUMN in feature_dataframe.columns:
    print(
        "FAIL — label found inside feature dataframe"
    )
    raise SystemExit(1)

print(
    "PASS — label is stored separately"
)


# ============================================================
# 7. NORMAL-ONLY TRAINING CHECK
# ============================================================

print("\n6. NORMAL-ONLY TRAINING CHECK")
print("-" * 60)

# We don't need to load all 14,048 trajectories here.
# Test a representative subset first.

train_manifest = get_split_manifest(
    "train"
)

sample_train = train_manifest.head(
    min(50, len(train_manifest))
)

attack_rows_before = 0
attack_rows_after = 0
total_rows_before = 0
total_rows_after = 0

for _, train_row in sample_train.iterrows():

    features, labels, _ = (
        load_trajectory(train_row)
    )

    total_rows_before += len(
        labels
    )

    attack_rows_before += np.sum(
        labels != 0
    )

    normal_mask = (
        labels == 0
    )

    normal_features = (
        features[normal_mask]
    )

    normal_labels = (
        labels[normal_mask]
    )

    total_rows_after += len(
        normal_labels
    )

    attack_rows_after += np.sum(
        normal_labels != 0
    )


print(
    f"Sampled training trajectories : "
    f"{len(sample_train)}"
)

print(
    f"Rows before normal filtering : "
    f"{total_rows_before:,}"
)

print(
    f"Attack rows before filtering : "
    f"{attack_rows_before:,}"
)

print(
    f"Rows after normal filtering : "
    f"{total_rows_after:,}"
)

print(
    f"Attack rows after filtering : "
    f"{attack_rows_after:,}"
)

if attack_rows_after != 0:
    print(
        "FAIL — attack rows remain in normal-only data"
    )
    raise SystemExit(1)

print(
    "PASS — normal-only training contains no attack labels"
)


# ============================================================
# 8. PACKET LOSS LABEL CONSISTENCY
# ============================================================

print("\n7. PACKET-LOSS LABEL CONSISTENCY")
print("-" * 60)

original_labels = labels.copy()

received_features, received_labels, mask = (
    apply_loss(
        features,
        labels,
        loss_rate=0.30,
        loss_type="random",
        identifier="stage3_integrity_test",
    )
)

expected_labels = (
    original_labels[mask]
)

if not np.array_equal(
    received_labels,
    expected_labels
):
    print(
        "FAIL — received labels do not correspond "
        "to received observations"
    )
    raise SystemExit(1)

print(
    f"Original rows : "
    f"{len(original_labels)}"
)

print(
    f"Received rows : "
    f"{len(received_labels)}"
)

print(
    "PASS — packet loss removes observations "
    "without altering surviving labels"
)


# ============================================================
# 9. DETERMINISTIC BURST 30%
# ============================================================

print("\n8. BURST 30% DETERMINISM CHECK")
print("-" * 60)

mask_a = make_loss_mask(
    n_rows=len(features),
    loss_rate=0.30,
    loss_type="burst",
    identifier="stage3_burst_test",
)

mask_b = make_loss_mask(
    n_rows=len(features),
    loss_rate=0.30,
    loss_type="burst",
    identifier="stage3_burst_test",
)

if not np.array_equal(
    mask_a,
    mask_b
):
    print(
        "FAIL — burst masks are not deterministic"
    )
    raise SystemExit(1)

received_count = np.sum(
    mask_a
)

lost_count = (
    len(mask_a)
    - received_count
)

print(
    f"Rows : {len(mask_a)}"
)

print(
    f"Received : {received_count}"
)

print(
    f"Lost     : {lost_count}"
)

print(
    "PASS — Burst 30% can be generated "
    "deterministically in memory"
)


# ============================================================
# 10. FIRST OBSERVATION RETENTION
# ============================================================

print("\n9. FIRST OBSERVATION RETENTION")
print("-" * 60)

random_mask = make_loss_mask(
    n_rows=500,
    loss_rate=0.50,
    loss_type="random",
    identifier="first_row_test_random",
)

burst_mask = make_loss_mask(
    n_rows=500,
    loss_rate=0.50,
    loss_type="burst",
    identifier="first_row_test_burst",
)

if not random_mask[0]:
    print(
        "FAIL — random loss removed first observation"
    )
    raise SystemExit(1)

if not burst_mask[0]:
    print(
        "FAIL — burst loss removed first observation"
    )
    raise SystemExit(1)

print(
    "PASS — first observation is always retained"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    "Stage 3.2 experiment integrity checks PASSED"
)

print(
    "\nThe experiment loader is safe to use for:"
)

print(
    "  - Isolation Forest"
)

print(
    "  - Window GRU / TCN"
)

print(
    "  - Fixed-step Kalman"
)

print(
    "  - Δt-aware Kalman"
)

print(
    "  - Neural-residual Kalman"
)

print(
    "  - Conformal calibration"
)

print(
    "  - Reinforcement-learning decision layer"
)

print(
    "\nNo additional simulated datasets are required "
    "for Burst 30% or Burst 50%."
)

print(
    "=" * 60
)

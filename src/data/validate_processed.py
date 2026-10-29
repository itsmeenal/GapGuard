from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectories"
)

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_manifest.csv"
)


# =========================================================
# EXPECTED COLUMNS
# =========================================================

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

EXPECTED_EXTRA_COLUMNS = [
    "label",
    "route",
    "flight_id",
    "file_label",
    "split",
]


EXPECTED_COLUMNS = (
    FEATURE_COLUMNS
    + EXPECTED_EXTRA_COLUMNS
)


# =========================================================
# LOAD MANIFEST
# =========================================================

manifest = pd.read_csv(MANIFEST_FILE)

print("====================================")
print("STEP 2.3 — PROCESSED DATA VALIDATION")
print("====================================")

print(f"Manifest rows: {len(manifest)}")


# =========================================================
# FIND PROCESSED FILES
# =========================================================

processed_files = list(
    PROCESSED_DIR.glob("*.csv")
)

print(
    f"Processed trajectory files: "
    f"{len(processed_files)}"
)


# =========================================================
# CHECK 1 — FILE COUNT
# =========================================================

expected_files = len(manifest)
actual_files = len(processed_files)

if actual_files == expected_files:
    print("PASS — file count matches manifest")
else:
    print(
        "FAIL — file count mismatch:",
        actual_files,
        "vs",
        expected_files
    )


# =========================================================
# CHECK 2 — COLUMN STRUCTURE
# =========================================================

column_errors = []

for path in processed_files:

    try:
        df = pd.read_csv(path, nrows=1)

        if list(df.columns) != EXPECTED_COLUMNS:

            column_errors.append(
                (
                    path.name,
                    list(df.columns)
                )
            )

    except Exception as e:

        column_errors.append(
            (
                path.name,
                str(e)
            )
        )


if len(column_errors) == 0:

    print(
        "PASS — all files have the expected columns"
    )

else:

    print(
        "FAIL — files with column problems:",
        len(column_errors)
    )

    for item in column_errors[:10]:
        print(item)


# =========================================================
# CHECK 3 — SAMPLE NUMERIC VALIDATION
# =========================================================

rng = np.random.default_rng(42)

sample_size = min(
    500,
    len(processed_files)
)

sample_files = rng.choice(
    processed_files,
    size=sample_size,
    replace=False
)

numeric_errors = []
nan_errors = []
inf_errors = []


for path in sample_files:

    df = pd.read_csv(path)

    # ---------------------------------------------
    # Numeric columns
    # ---------------------------------------------

    for col in FEATURE_COLUMNS:

        if not pd.api.types.is_numeric_dtype(
            df[col]
        ):

            numeric_errors.append(
                (path.name, col)
            )

    # ---------------------------------------------
    # NaN
    # ---------------------------------------------

    if df[FEATURE_COLUMNS].isna().any().any():

        nan_errors.append(path.name)

    # ---------------------------------------------
    # Infinite values
    # ---------------------------------------------

    values = df[FEATURE_COLUMNS].to_numpy()

    if not np.isfinite(values).all():

        inf_errors.append(path.name)


print(
    f"\nChecked {sample_size} random trajectories "
    f"for numeric validity"
)

print(
    "NaN-containing files:",
    len(nan_errors)
)

print(
    "Infinite-value files:",
    len(inf_errors)
)

print(
    "Non-numeric feature errors:",
    len(numeric_errors)
)


# =========================================================
# CHECK 4 — TIME / DT CONSISTENCY
# =========================================================

dt_errors = []
non_increasing_time = []

for path in sample_files:

    df = pd.read_csv(path)

    expected_dt = (
        df["t_elapsed_sec"]
        .diff()
        .fillna(0)
    )

    actual_dt = df["dt"]

    max_error = np.abs(
        actual_dt - expected_dt
    ).max()

    if max_error > 1e-6:

        dt_errors.append(
            (
                path.name,
                float(max_error)
            )
        )

    time_diff = (
        df["t_elapsed_sec"]
        .diff()
        .iloc[1:]
    )

    if (time_diff <= 0).any():

        non_increasing_time.append(
            path.name
        )


print("\nTime validation:")
print(
    "dt mismatch files:",
    len(dt_errors)
)

print(
    "non-increasing time files:",
    len(non_increasing_time)
)


# =========================================================
# CHECK 5 — LABEL DISTRIBUTION
# =========================================================

label_counts = {}

for path in sample_files:

    df = pd.read_csv(path)

    for label, count in (
        df["label"]
        .value_counts()
        .items()
    ):

        label_counts[int(label)] = (
            label_counts.get(int(label), 0)
            + int(count)
        )


print("\nSample label distribution:")

for label in sorted(label_counts):

    print(
        f"label {label}: "
        f"{label_counts[label]:,} rows"
    )


# =========================================================
# CHECK 6 — SPLIT DISTRIBUTION
# =========================================================

summary = pd.read_csv(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processing_summary.csv"
)

print("\nProcessed trajectory split counts:")

print(
    summary["split"]
    .value_counts()
    .sort_index()
)


# =========================================================
# FINAL STATUS
# =========================================================

all_pass = (
    actual_files == expected_files
    and len(column_errors) == 0
    and len(numeric_errors) == 0
    and len(nan_errors) == 0
    and len(inf_errors) == 0
    and len(dt_errors) == 0
    and len(non_increasing_time) == 0
)


print("\n====================================")

if all_pass:

    print("ALL CHECKS PASSED ✅")

else:

    print("SOME CHECKS FAILED ⚠️")

print("====================================")
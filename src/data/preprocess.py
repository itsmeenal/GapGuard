from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectories"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# DETECTOR INPUT FEATURES
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


# Ground-truth column.
# This is stored separately and NEVER passed to the detector.

LABEL_COLUMN = "label"


# =========================================================
# LOAD MANIFEST
# =========================================================

manifest = pd.read_csv(MANIFEST_FILE)

print("Loaded clean manifest")
print(f"Manifest rows: {len(manifest)}")


# =========================================================
# BASIC VALIDATION
# =========================================================

required_manifest_columns = [
    "route",
    "flight_id",
    "file_label",
    "split",
    "csv_path",
]

missing_manifest_columns = [
    col
    for col in required_manifest_columns
    if col not in manifest.columns
]

if missing_manifest_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_manifest_columns}"
    )


# =========================================================
# PROCESS EACH TRAJECTORY
# =========================================================

processed = 0
failed = 0

summary = []


for _, row in manifest.iterrows():

    csv_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "dataset"
        / row["csv_path"]
    )

    # -----------------------------------------------------
    # Check file exists
    # -----------------------------------------------------

    if not csv_path.exists():

        print(
            f"WARNING: file not found: {csv_path}"
        )

        failed += 1
        continue


    try:

        df = pd.read_csv(csv_path)


        # -------------------------------------------------
        # Check required columns
        # -------------------------------------------------

        required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            print(
                f"WARNING: missing columns in {csv_path.name}: "
                f"{missing_columns}"
            )

            failed += 1
            continue


        # -------------------------------------------------
        # Select only the columns we need
        # -------------------------------------------------

        clean_df = df[
            FEATURE_COLUMNS + [LABEL_COLUMN]
        ].copy()


        # -------------------------------------------------
        # Convert numeric columns
        # -------------------------------------------------

        for col in FEATURE_COLUMNS + [LABEL_COLUMN]:

            clean_df[col] = pd.to_numeric(
                clean_df[col],
                errors="coerce"
            )


        # -------------------------------------------------
        # Remove rows with missing detector inputs
        # -------------------------------------------------

        before_rows = len(clean_df)

        clean_df = clean_df.dropna(
            subset=FEATURE_COLUMNS + [LABEL_COLUMN]
        ).reset_index(drop=True)

        removed_rows = (
            before_rows - len(clean_df)
        )


        # -------------------------------------------------
        # Sort by elapsed time
        # -------------------------------------------------

        clean_df = clean_df.sort_values(
            "t_elapsed_sec"
        ).reset_index(drop=True)


        # -------------------------------------------------
        # Verify time is non-decreasing
        # -------------------------------------------------

        time_diff = (
            clean_df["t_elapsed_sec"]
            .diff()
            .iloc[1:]
        )

        if (time_diff <= 0).any():

            print(
                f"WARNING: non-increasing time in "
                f"{csv_path.name}"
            )

            failed += 1
            continue


        # -------------------------------------------------
        # Verify dt matches elapsed-time difference
        # -------------------------------------------------

        expected_dt = (
            clean_df["t_elapsed_sec"]
            .diff()
            .fillna(0)
        )

        dt_error = np.abs(
            clean_df["dt"] - expected_dt
        )

        max_dt_error = dt_error.max()

        if max_dt_error > 1e-6:

            print(
                f"WARNING: dt mismatch in "
                f"{csv_path.name}; "
                f"max error={max_dt_error}"
            )

            failed += 1
            continue


        # -------------------------------------------------
        # Add metadata
        # -------------------------------------------------

        clean_df["route"] = row["route"]
        clean_df["flight_id"] = str(row["flight_id"])
        clean_df["file_label"] = int(row["file_label"])
        clean_df["split"] = row["split"]


        # -------------------------------------------------
        # Save processed trajectory
        # -------------------------------------------------

        output_name = (
            f"{row['route']}_"
            f"{row['flight_id']}_"
            f"v{row['file_label']}.csv"
        )

        output_path = OUTPUT_DIR / output_name

        clean_df.to_csv(
            output_path,
            index=False
        )


        summary.append({
            "route": row["route"],
            "flight_id": str(row["flight_id"]),
            "file_label": int(row["file_label"]),
            "split": row["split"],
            "rows": len(clean_df),
            "removed_rows": removed_rows,
            "output_file": output_name,
        })


        processed += 1


    except Exception as e:

        print(
            f"ERROR processing {csv_path.name}: {e}"
        )

        failed += 1


# =========================================================
# SAVE PROCESSING SUMMARY
# =========================================================

summary_df = pd.DataFrame(summary)

summary_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "processing_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n====================================")
print("PREPROCESSING COMPLETE")
print("====================================")

print(f"Successfully processed: {processed}")
print(f"Failed: {failed}")

if len(summary_df) > 0:

    print(
        f"Total processed rows: "
        f"{summary_df['rows'].sum()}"
    )

    print(
        f"Total removed rows: "
        f"{summary_df['removed_rows'].sum()}"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nProcessing summary:")
print(summary_path)
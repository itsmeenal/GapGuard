from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_manifest.csv"
)


# ---------------------------------------------------------
# Find all manifest files
# ---------------------------------------------------------

manifest_files = list(RAW_DATASET.glob("data_*_seq/split_manifest.csv"))

print(f"Found {len(manifest_files)} manifest files")


if len(manifest_files) == 0:
    raise FileNotFoundError(
        "No split_manifest.csv files were found."
    )


# ---------------------------------------------------------
# Read manifests
# ---------------------------------------------------------

all_manifests = []

for manifest_path in manifest_files:

    df = pd.read_csv(manifest_path)

    # Add route from parent folder name
    route = manifest_path.parent.name.replace("data_", "").replace("_seq", "")

    df["route"] = route

    all_manifests.append(df)


manifest = pd.concat(
    all_manifests,
    ignore_index=True
)


# ---------------------------------------------------------
# Convert important columns
# ---------------------------------------------------------

manifest["flight_id"] = manifest["flight_id"].astype(str)
manifest["file_label"] = manifest["file_label"].astype(int)

manifest["route"] = manifest["route"].str.lower()
manifest["split"] = manifest["split"].str.lower()

manifest["csv_path"] = manifest["csv_path"].astype(str)


# ---------------------------------------------------------
# Remove exact duplicate manifest entries
# ---------------------------------------------------------

before = len(manifest)

manifest = manifest.drop_duplicates(
    subset=["route", "flight_id", "csv_path"]
)

after = len(manifest)

print(f"Removed exact duplicate manifest rows: {before - after}")


# ---------------------------------------------------------
# Identify the known inconsistent PPS position file
# ---------------------------------------------------------

known_bad_file = (
    "data_pps_ready/ppspos/modpos_pps_3e54a0d8.csv"
)

bad_mask = manifest["csv_path"].str.replace(
    "\\",
    "/",
    regex=False
) == known_bad_file

print(
    f"Known inconsistent PPS position file found: "
    f"{bad_mask.sum()}"
)


# ---------------------------------------------------------
# Exclude the known problematic trajectory version
#
# IMPORTANT:
# We are NOT modifying the original dataset.
# We simply exclude this inconsistent file from
# our experimental manifest.
# ---------------------------------------------------------

manifest = manifest.loc[~bad_mask].copy()


# ---------------------------------------------------------
# Check split consistency at flight level
# ---------------------------------------------------------

split_counts = (
    manifest
    .groupby(["route", "flight_id"])["split"]
    .nunique()
)

bad_split_flights = split_counts[split_counts > 1]

print(
    f"Flights appearing in multiple splits: "
    f"{len(bad_split_flights)}"
)


if len(bad_split_flights) > 0:
    print(bad_split_flights.head(20))
    raise ValueError(
        "Flight-level split leakage detected."
    )


# ---------------------------------------------------------
# Save clean manifest
# ---------------------------------------------------------

manifest = manifest.sort_values(
    ["route", "flight_id", "file_label", "csv_path"]
).reset_index(drop=True)


manifest.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n==============================")
print("CLEAN MANIFEST CREATED")
print("==============================")

print(f"Rows: {len(manifest)}")
print(f"Flights: {manifest[['route', 'flight_id']].drop_duplicates().shape[0]}")

print("\nRoutes:")
print(
    manifest["route"]
    .value_counts()
    .sort_index()
)

print("\nSplits:")
print(
    manifest["split"]
    .value_counts()
    .sort_index()
)

print("\nOutput:")
print(OUTPUT_FILE)
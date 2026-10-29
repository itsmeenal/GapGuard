from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/raw/dataset")

routes = ["ceb", "dvo", "ilo", "mph", "pps"]

M = pd.concat(
    [
        pd.read_csv(
            ROOT / f"data_{r}_seq" / "split_manifest.csv"
        ).assign(route=r)
        for r in routes
    ],
    ignore_index=True
)

M["flight_id"] = M["flight_id"].astype(str)
M["file_label"] = M["file_label"].astype(int)

# ============================================================
# 1. Find duplicate manifest row
# ============================================================

print("=" * 70)
print("1. DUPLICATE MANIFEST ROW")
print("=" * 70)

dup_mask = M.duplicated(
    ["route", "flight_id", "file_label"],
    keep=False
)

print(M.loc[
    dup_mask,
    ["route", "flight_id", "file_label", "split", "csv_path"]
].to_string(index=False))


# ============================================================
# 2. Verify dt against t_elapsed_sec
# ============================================================

print("\n" + "=" * 70)
print("2. VERIFY dt = diff(t_elapsed_sec)")
print("=" * 70)

auth = M[M["file_label"] == 0].copy()

rng = np.random.default_rng(42)

sample_n = min(500, len(auth))

sample = auth.sample(
    sample_n,
    random_state=42
)

max_abs_error = 0.0
mean_abs_error = []
bad_files = 0
negative_dt_files = 0
total_points = 0

examples = []

for _, row in sample.iterrows():

    path = ROOT / row["csv_path"]

    df = pd.read_csv(path)

    t = df["t_elapsed_sec"].to_numpy(dtype=float)
    dt = df["dt"].to_numpy(dtype=float)

    if len(df) < 2:
        continue

    expected = np.diff(t)

    # dt should normally have the same length as the data.
    # Compare dt[1:] with differences between consecutive timestamps.
    if len(dt) != len(t):
        bad_files += 1
        continue

    actual = dt[1:]

    err = np.abs(actual - expected)

    max_err = float(np.max(err))
    mean_err = float(np.mean(err))

    max_abs_error = max(max_abs_error, max_err)
    mean_abs_error.append(mean_err)

    total_points += len(err)

    if max_err > 1e-6:
        bad_files += 1

        if len(examples) < 5:
            examples.append(
                (
                    row["route"],
                    row["flight_id"],
                    row["csv_path"],
                    max_err,
                    mean_err
                )
            )

    if np.any(dt[1:] <= 0):
        negative_dt_files += 1


print("files checked:", sample_n)
print("files with dt/t mismatch:", bad_files)
print("files with non-positive dt:", negative_dt_files)
print("max absolute error:", max_abs_error)

if mean_abs_error:
    print(
        "mean file-level absolute error:",
        float(np.mean(mean_abs_error))
    )

print("total dt points compared:", total_points)

if examples:
    print("\nExample mismatches:")
    for x in examples:
        print(x)


# ============================================================
# 3. Check dt=0 convention for first row
# ============================================================

print("\n" + "=" * 70)
print("3. FIRST-ROW dt CHECK")
print("=" * 70)

zero_count = 0
first_values = []

for _, row in sample.head(20).iterrows():

    df = pd.read_csv(ROOT / row["csv_path"])

    first_dt = float(df["dt"].iloc[0])

    first_values.append(first_dt)

    if abs(first_dt) < 1e-9:
        zero_count += 1

print("first 20 dt values:", first_values)
print("first-row dt == 0 count:", zero_count, "/ 20")


# ============================================================
# 4. Check duplicate CSV paths in manifest
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATE CSV PATHS")
print("=" * 70)

path_counts = M["csv_path"].value_counts()

print(
    "CSV paths appearing more than once:",
    int((path_counts > 1).sum())
)

if (path_counts > 1).any():
    print(
        path_counts[path_counts > 1]
        .head(10)
        .to_string()
    )
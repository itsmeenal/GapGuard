from pathlib import Path
import random
import numpy as np
import pandas as pd

ROOT = Path("data/raw/dataset")
random.seed(42)

def fname(x):
    return Path(str(x).replace("\\", "/")).name

index = {p.name: p for p in ROOT.rglob("*.csv")}
pool = [p for p in ROOT.rglob("*.csv") if any(s.endswith("_ready") for s in p.parts)]

print("=" * 60, "\n1. MANIFEST (Cebu)")
m = pd.read_csv(ROOT / "data_ceb_seq" / "split_manifest.csv")
print(m.head(8).to_string())
print(m["split"].value_counts().to_string())
print(m["file_label"].value_counts().to_string())
print("unique flights:", m["flight_id"].nunique())
print("versions per flight:", m.groupby("flight_id").size().value_counts().to_dict())

print("=" * 60, "\n2. ONE FLIGHT, ALL VERSIONS")
fid = m["flight_id"].iloc[0]
dfs = {}
for _, r in m[m["flight_id"] == fid].iterrows():
    p = index.get(fname(r["csv_path"]))
    if p is None:
        print("NOT FOUND:", r["csv_path"]); continue
    d = pd.read_csv(p)
    dfs[str(r["file_label"])] = d
    print(f"{r['file_label']:>8} | {p.name} | rows={len(d)} | labels={d['label'].value_counts().to_dict()}")

auth_key = next((k for k in dfs if "auth" in k.lower()), None)
cols = ["latitude", "longitude", "altitude_meters", "speed_kmh", "verticalSpeed_ms", "heading"]
if auth_key:
    a = dfs[auth_key]
    for k, d in dfs.items():
        if k == auth_key:
            continue
        if len(d) != len(a):
            print(k, "-> length differs from authentic:", len(d), "vs", len(a)); continue
        changed = np.zeros(len(d), bool)
        for c in cols:
            diff = ~np.isclose(a[c].values, d[c].values, equal_nan=True)
            changed |= diff
            if diff.any():
                idx = np.where(diff)[0]
                print(f"{k:>8} | {c:<17} differs in {diff.sum():4d}/{len(d)} rows (first={idx[0]}, last={idx[-1]})")
        lab_diff = d["label"].astype(str).values != a["label"].astype(str).values
        print(f"{k:>8} | rows changed={changed.sum()} | label differs from authentic={lab_diff.sum()} | overlap={(changed & lab_diff).sum()}")

print("=" * 60, "\n3. TIMING ACROSS 100 RANDOM FILES")
stats = []
for p in random.sample(pool, 100):
    d = pd.read_csv(p, usecols=["t_elapsed_sec", "dt"])
    td = d["t_elapsed_sec"].diff().dropna()
    stats.append(dict(rows=len(d), dt_unique=d["dt"].nunique(), dt_min=d["dt"].min(),
                      dt_med=d["dt"].median(), dt_max=d["dt"].max(),
                      td_unique=td.round(3).nunique(), td_min=td.min(), td_med=td.median(), td_max=td.max()))
s = pd.DataFrame(stats)
print(s.describe().to_string())
print("files with constant dt:", int((s.dt_unique <= 1).sum()), "of", len(s))

print("=" * 60, "\n4. RAW FOLDER")
for route in ["ceb", "pps"]:
    p = next((ROOT / "raw" / f"data_{route}").glob("*.csv"))
    d = pd.read_csv(p, nrows=5)
    print(p.name, list(d.columns))
    print(d.to_string(index=False))

print("=" * 60, "\n5. FLIGHT LENGTHS")
lens = [sum(1 for _ in open(p, "rb")) - 1 for p in random.sample(pool, 200)]
print(pd.Series(lens).describe().to_string())
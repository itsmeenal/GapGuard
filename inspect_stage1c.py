from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/raw/dataset")
routes = ["ceb", "dvo", "ilo", "mph", "pps"]
M = pd.concat([pd.read_csv(ROOT / f"data_{r}_seq" / "split_manifest.csv").assign(route=r)
               for r in routes], ignore_index=True)
M["flight_id"] = M["flight_id"].astype(str)
M["file_label"] = M["file_label"].astype(int)

# --- diagnostics ---
per_fid_routes = M.groupby("flight_id")["route"].nunique()
print("flight_ids that appear in more than one route:", int((per_fid_routes > 1).sum()))
dup = M.duplicated(["route", "flight_id", "file_label"]).sum()
print("duplicate (route, flight_id, label) rows:", int(dup))
print("versions per (route, flight):", M.groupby(["route", "flight_id"]).size().value_counts().to_dict())
print("(route, flight) in more than one split:",
      int((M.groupby(["route", "flight_id"])["split"].nunique() > 1).sum()))
print("\nFLIGHTS PER ROUTE / SPLIT")
print(M[M.file_label == 0].groupby(["route", "split"]).size().unstack().to_string())

cols = ["latitude", "longitude", "altitude_meters", "speed_kmh", "verticalSpeed_ms", "heading"]
keys = (M[M.file_label == 0][["route", "flight_id"]]
        .drop_duplicates().sample(300, random_state=42).values.tolist())

def load(route, fid):
    sub = M[(M.route == route) & (M.flight_id == fid)].drop_duplicates("file_label")
    out = {}
    for _, r in sub.iterrows():
        out[int(r["file_label"])] = pd.read_csv(str(ROOT / r["csv_path"]))
    return out

rows, gaps, bad_time, skipped = [], [], 0, 0
for route, fid in keys:
    v = load(route, fid)
    if not all(k in v for k in (0, 1, 2, 3)):
        skipped += 1
        continue
    a = v[0]
    t = a["t_elapsed_sec"].values
    bad_time += int(not (np.diff(t) > 0).all())
    gaps.extend(np.diff(t))
    for k in (1, 2, 3):
        d = v[k]
        if len(d) != len(a):
            skipped += 1
            continue
        lab = (d["label"].values == k)
        changed = np.zeros(len(d), bool)
        which = []
        for c in cols:
            m = ~np.isclose(a[c].values, d[c].values, equal_nan=True)
            if m.any():
                which.append(c)
            changed |= m
        idx = np.where(lab)[0]
        runs = int((np.diff(idx) > 1).sum() + 1) if len(idx) else 0
        coslat = np.cos(np.radians(a["latitude"].values))
        dpos = np.hypot((d["latitude"].values - a["latitude"].values) * 111000,
                        (d["longitude"].values - a["longitude"].values) * 111000 * coslat)
        rows.append(dict(
            attack=k, n=len(d), attack_rows=int(lab.sum()), runs=runs,
            start_frac=idx[0] / len(d) if len(idx) else np.nan,
            changed_rows=int(changed.sum()),
            changed_not_labeled=int((changed & ~lab).sum()),
            labeled_not_changed=int((lab & ~changed).sum()),
            alt_m=np.abs(d["altitude_meters"].values - a["altitude_meters"].values)[lab].mean() if lab.any() else np.nan,
            spd_kmh=np.abs(d["speed_kmh"].values - a["speed_kmh"].values)[lab].mean() if lab.any() else np.nan,
            pos_m=dpos[lab].mean() if lab.any() else np.nan,
            cols_changed="|".join(which)))

R = pd.DataFrame(rows)
print("\nskipped flights/versions:", skipped, "| rows analysed:", len(R))
print("\nATTACK SUMMARY (per attack type)")
print(R.groupby("attack")[["attack_rows", "runs", "start_frac", "changed_not_labeled",
                           "labeled_not_changed", "alt_m", "spd_kmh", "pos_m"]].describe().T.to_string())
print("\nCOLUMNS CHANGED:")
print(R.groupby("attack")["cols_changed"].value_counts().to_string())

g = np.array(gaps)
print("\nNATURAL GAPS in authentic flights (seconds)")
print(pd.Series(g).quantile([.5, .9, .95, .99, .999]).to_string())
print("share >10s:", (g > 10).mean(), "| >30s:", (g > 30).mean(), "| >60s:", (g > 60).mean())
print("flights with non-increasing time:", bad_time)
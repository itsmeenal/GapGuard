from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path("data/raw")
files = [p for p in DATA_DIR.rglob("*") if p.is_file()]
csvs = [p for p in files if p.suffix.lower() == ".csv"]
print("Total files:", len(files), "| CSV files:", len(csvs))

# 1. How many CSVs per folder (reveals authentic / modified / poison / windows)
print("\nCSV COUNT PER FOLDER")
by_dir = Counter(str(p.parent.relative_to(DATA_DIR)) for p in csvs)
for d, n in sorted(by_dir.items())[:80]:
    print(f"{n:6d}  {d}")

# 2. One sample file per folder: columns and 2 rows
print("\nSCHEMA PER FOLDER")
first = {}
for p in csvs:
    first.setdefault(p.parent, p)
for folder, p in list(first.items())[:30]:
    try:
        df = pd.read_csv(p, nrows=2)
        print("\n", folder.relative_to(DATA_DIR))
        print(list(df.columns))
    except Exception as e:
        print("Could not read", p, e)

# 3. Twin check: same filename in more than one folder?
names = Counter(p.name for p in csvs)
twins = [n for n, c in names.items() if c > 1]
print("\nFilenames appearing in 2+ folders:", len(twins))
print("Examples:", twins[:5])

# 4. Timing check on one sample file
p = csvs[0]
df = pd.read_csv(p)
tcols = [c for c in df.columns if "time" in c.lower()]
print("\nTiming sample:", p.name, "| time-like columns:", tcols)
if tcols:
    t = pd.to_numeric(df[tcols[0]], errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(df[tcols[0]], errors="coerce").astype("int64") / 1e9
    dt = t.diff().dropna()
    print(dt.describe())
    print("Distinct gap values:", dt.nunique())

# 5. If twins exist, count rows that differ between two versions
if twins:
    paths = [q for q in csvs if q.name == twins[0]]
    a, b = pd.read_csv(paths[0]), pd.read_csv(paths[1])
    print("\nTwin compare:", paths[0], "vs", paths[1])
    print("Rows:", len(a), len(b))
    common = [c for c in a.columns if c in b.columns and a[c].dtype != object]
    if len(a) == len(b):
        diff = (a[common] != b[common]).any(axis=1)
        print("Rows that differ:", int(diff.sum()), "of", len(a))
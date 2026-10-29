from pathlib import Path
import pandas as pd

ROOT = Path("data/raw/dataset")

files = [
    "data_pps_ready/ppsauth/auth_pps_3e54a0d8.csv",
    "data_pps_ready/ppspos/modpos_pps_3e54a0d8.csv",
    "data_pps_ready/ppsalt/modalt_pps_3e54a0d8.csv",
    "data_pps_ready/ppsspd/modspd_pps_3e54a0d8.csv",
]

for f in files:
    path = ROOT / f
    df = pd.read_csv(path)

    print("\n", f)
    print("rows:", len(df))
    print("labels:", df["label"].value_counts().to_dict())
    print(
        "first 5 values:",
        df[
            [
                "latitude",
                "longitude",
                "altitude_meters",
                "speed_kmh",
                "label"
            ]
        ].head().to_dict("records")
    )
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAJECTORY_DIR = PROJECT_ROOT / "data" / "processed" / "trajectories"

REQUIRED_COLUMNS = [
    "t_elapsed_sec",
    "dt",
    "latitude",
    "longitude",
    "altitude_meters",
    "speed_kmh",
    "verticalSpeed_ms",
    "heading",
]

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1884
TOPIC = "gapguard/adsb/telemetry"


def find_trajectory():
    print("[MQTT] Looking for processed ADS-B trajectory...")
    print(f"[MQTT] Trajectory directory: {TRAJECTORY_DIR}")

    if not TRAJECTORY_DIR.exists():
        raise FileNotFoundError(
            f"Trajectory directory not found:\n{TRAJECTORY_DIR}"
        )

    # We already know processed files use the _v0 naming convention.
    candidates = sorted(TRAJECTORY_DIR.glob("*_v0.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No processed trajectory (*_v0.csv) found in:\n{TRAJECTORY_DIR}"
        )

    print(f"[MQTT] Found {len(candidates)} trajectory files.")

    # Use the first authentic version for the live demonstration.
    selected = candidates[0]

    print(f"[MQTT] Selected trajectory:")
    print(f"[MQTT] {selected}")

    return selected


def main():

    trajectory_path = find_trajectory()

    df = pd.read_csv(trajectory_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    # Remove invalid rows.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) == 0:
        raise ValueError("Selected trajectory contains no valid rows.")

    # Deterministic 30% packet-loss simulation.
    rng = np.random.default_rng(42)

    keep = rng.random(len(df)) >= 0.30

    # Always retain the first observation.
    keep[0] = True

    df = df.loc[keep].reset_index(drop=True)

    print(f"[MQTT] Rows available for streaming: {len(df)}")
    print("[MQTT] Simulated packet loss: 30%")
    print("[MQTT] Connecting to AMQTT broker...")
    print(f"[MQTT] Broker: {MQTT_HOST}:{MQTT_PORT}")

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv311
    )

    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()

    print("[MQTT] Publisher connected.")
    print(f"[MQTT] Publishing to: {TOPIC}")
    print()

    try:
        for i, row in df.iterrows():

            payload = {
                "t_elapsed_sec": float(row["t_elapsed_sec"]),
                "dt": float(row["dt"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "altitude_meters": float(row["altitude_meters"]),
                "speed_kmh": float(row["speed_kmh"]),
                "verticalSpeed_ms": float(row["verticalSpeed_ms"]),
                "heading": float(row["heading"]),
            }

            client.publish(
                TOPIC,
                json.dumps(payload),
                qos=0
            )

            print(
                f"[PUBLISH] {i + 1:03d}/{len(df):03d} "
                f"t={payload['t_elapsed_sec']:.2f}s"
            )

            time.sleep(0.03)

    finally:
        client.loop_stop()
        client.disconnect()

    print()
    print("[MQTT] Publisher finished successfully.")


if __name__ == "__main__":
    main()
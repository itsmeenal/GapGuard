"""
GapGuard Stage 4.10
Live synthetic attack demonstration over MQTT.

This demo does NOT modify:
- the validated detector
- frozen thresholds
- offline experiment results
- processed dataset

It only creates a temporary synthetic anomaly in one MQTT message.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

from src.streaming.detector import GapGuardDetector


HOST = "127.0.0.1"
PORT = 1884

TELEMETRY_TOPIC = "gapguard/adsb/telemetry"
ALERT_TOPIC = "gapguard/alerts"

NUMBER_OF_NORMAL_MESSAGES = 12

# Inject the anomaly after a normal message.
ATTACK_INDEX = 6

# Large altitude displacement.
# This is intentionally obvious and synthetic.
ALTITUDE_OFFSET_METERS = 5000.0


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAJECTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "trajectories"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "system"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def find_demo_trajectory():
    files = sorted(
        TRAJECTORY_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No processed trajectories found in {TRAJECTORY_DIR}"
        )

    return files[0]


def load_messages(path):
    df = pd.read_csv(path)

    required_columns = [
        "t_elapsed_sec",
        "dt",
        "latitude",
        "longitude",
        "altitude_meters",
        "speed_kmh",
        "verticalSpeed_ms",
        "heading",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        required_columns
    ].copy()

    # Use the beginning of the trajectory.
    # The detector therefore starts with a small/high-quality gap.
    return df.head(
        NUMBER_OF_NORMAL_MESSAGES
    )


class AlertListener:

    def __init__(self):
        self.alerts = []

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = (
            self.on_connect
        )

        self.client.on_message = (
            self.on_message
        )

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ):

        print(
            f"[ALERT LISTENER] Connected "
            f"(reason={reason_code})"
        )

        client.subscribe(
            ALERT_TOPIC,
            qos=0,
        )

        print(
            f"[ALERT LISTENER] "
            f"Subscribed: {ALERT_TOPIC}"
        )

    def on_message(
        self,
        client,
        userdata,
        message,
    ):

        try:

            payload = json.loads(
                message.payload.decode(
                    "utf-8"
                )
            )

            self.alerts.append(
                payload
            )

            print()
            print("=" * 80)
            print("🚨 GAPGUARD ALERT RECEIVED")
            print("=" * 80)

            print(
                f"Timestamp : "
                f"{payload.get('timestamp', 0):.2f}s"
            )

            print(
                f"Gap       : "
                f"{payload.get('gap_seconds', 0):.2f}s"
            )

            print(
                f"Severity  : "
                f"{payload.get('severity', 0):.3f}"
            )

            print(
                f"Evidence  : "
                f"{payload.get('evidence', 'UNKNOWN').upper()}"
            )

            print(
                f"Threshold : "
                f"{payload.get('threshold', 0):.3f}"
            )

            print(
                f"Decision  : "
                f"{payload.get('decision', 'UNKNOWN')}"
            )

            print("=" * 80)

        except Exception as exc:

            print(
                f"[ALERT LISTENER] "
                f"Error: {exc}"
            )


def main():

    print("=" * 80)
    print("STAGE 4.10 — LIVE SYNTHETIC ATTACK DEMONSTRATION")
    print("=" * 80)

    trajectory_path = (
        find_demo_trajectory()
    )

    print(
        f"Trajectory : {trajectory_path.name}"
    )

    df = load_messages(
        trajectory_path
    )

    print(
        f"Messages   : {len(df)}"
    )

    print(
        f"Attack     : altitude injection"
    )

    print(
        f"Injection  : +{ALTITUDE_OFFSET_METERS:.0f} meters"
    )

    print()

    # ------------------------------------------------------------------
    # ALERT LISTENER
    # ------------------------------------------------------------------

    listener = AlertListener()

    listener.client.connect(
        HOST,
        PORT,
        60,
    )

    listener.client.loop_start()

    time.sleep(0.5)

    # ------------------------------------------------------------------
    # DETECTOR
    # ------------------------------------------------------------------

    detector = GapGuardDetector()

    publisher = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv311,
    )

    publisher.connect(
        HOST,
        PORT,
        60,
    )

    publisher.loop_start()

    time.sleep(0.5)

    published_messages = []

    # ------------------------------------------------------------------
    # STREAM NORMAL TELEMETRY
    # ------------------------------------------------------------------

    for index, row in df.iterrows():

        message = {
            "t_elapsed_sec":
                float(row["t_elapsed_sec"]),

            "dt":
                float(row["dt"]),

            "latitude":
                float(row["latitude"]),

            "longitude":
                float(row["longitude"]),

            "altitude_meters":
                float(row["altitude_meters"]),

            "speed_kmh":
                float(row["speed_kmh"]),

            "verticalSpeed_ms":
                float(row["verticalSpeed_ms"]),

            "heading":
                float(row["heading"]),
        }

        # --------------------------------------------------------------
        # SYNTHETIC ATTACK
        # --------------------------------------------------------------

        if index == ATTACK_INDEX:

            original_altitude = (
                message["altitude_meters"]
            )

            message[
                "altitude_meters"
            ] = (
                original_altitude
                + ALTITUDE_OFFSET_METERS
            )

            message[
                "synthetic_attack"
            ] = True

            print()
            print("=" * 80)
            print("⚠️  SYNTHETIC ATTACK INJECTED")
            print("=" * 80)

            print(
                f"Message index       : {index}"
            )

            print(
                f"Original altitude   : "
                f"{original_altitude:.2f} m"
            )

            print(
                f"Modified altitude   : "
                f"{message['altitude_meters']:.2f} m"
            )

            print(
                f"Injected difference : "
                f"+{ALTITUDE_OFFSET_METERS:.0f} m"
            )

            print("=" * 80)

        else:

            message[
                "synthetic_attack"
            ] = False

        # --------------------------------------------------------------
        # PUBLISH TO MQTT
        # --------------------------------------------------------------

        publisher.publish(
            TELEMETRY_TOPIC,
            json.dumps(message),
            qos=0,
        )

        published_messages.append(
            message
        )

        # Also process locally so the demonstration
        # does not depend on MQTT callback timing.
        result = detector.process(
            message
        )

        print(
            f"[STREAM] "
            f"index={index:02d} "
            f"| t={result['timestamp']:.2f}s "
            f"| gap={result['gap_seconds']:.2f}s "
            f"| severity={result['severity']:.3f} "
            f"| evidence={result['evidence'].upper()} "
            f"| decision={result['decision']}"
        )

        time.sleep(0.15)

    # ------------------------------------------------------------------
    # WAIT FOR ALERT CALLBACK
    # ------------------------------------------------------------------

    time.sleep(1.0)

    publisher.loop_stop()
    publisher.disconnect()

    listener.client.loop_stop()
    listener.client.disconnect()

    # ------------------------------------------------------------------
    # SAVE DEMO RESULT
    # ------------------------------------------------------------------

    attack_message = (
        published_messages[ATTACK_INDEX]
    )

    # Recalculate attack result independently
    # for a clean saved record.
    verification_detector = (
        GapGuardDetector()
    )

    verification_result = None

    for message in published_messages:

        verification_result = (
            verification_detector.process(
                message
            )
        )

    # The loop above ends at the final message,
    # so specifically replay up to the attack.
    verification_detector = (
        GapGuardDetector()
    )

    attack_result = None

    for index, message in enumerate(
        published_messages
    ):

        result = (
            verification_detector.process(
                message
            )
        )

        if index == ATTACK_INDEX:
            attack_result = result

    output = {

        "stage":
            "4.10",

        "description":
            "Live synthetic altitude attack demonstration",

        "trajectory":
            trajectory_path.name,

        "messages_published":
            len(published_messages),

        "attack_index":
            ATTACK_INDEX,

        "attack_type":
            "synthetic_altitude_injection",

        "altitude_offset_meters":
            ALTITUDE_OFFSET_METERS,

        "attack_message":
            attack_message,

        "attack_detection_result":
            attack_result,

        "mqtt_alerts_received":
            listener.alerts,

        "alert_received":
            len(listener.alerts) > 0,
    }

    output_path = (
        RESULTS_DIR
        / "live_attack_demo_result.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 80)
    print("STAGE 4.10 SUMMARY")
    print("=" * 80)

    print(
        f"Messages published : "
        f"{len(published_messages)}"
    )

    print(
        f"Attack injected    : YES"
    )

    print(
        f"Attack type        : "
        f"Synthetic altitude injection"
    )

    print(
        f"Altitude offset    : "
        f"+{ALTITUDE_OFFSET_METERS:.0f} m"
    )

    print(
        f"Attack severity    : "
        f"{attack_result['severity']:.3f}"
    )

    print(
        f"Evidence           : "
        f"{attack_result['evidence'].upper()}"
    )

    print(
        f"Decision            : "
        f"{attack_result['decision']}"
    )

    print(
        f"MQTT alert received: "
        f"{len(listener.alerts)}"
    )

    print(
        f"Saved              : "
        f"{output_path}"
    )

    if attack_result["decision"] == "ALERT":

        print()
        print(
            "STAGE 4.10 COMPLETE — "
            "SYNTHETIC ATTACK DETECTED"
        )

    else:

        print()
        print(
            "STAGE 4.10 COMPLETE — "
            "ATTACK DID NOT TRIGGER ALERT"
        )


if __name__ == "__main__":
    main()
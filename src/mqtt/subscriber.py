import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt

from src.streaming.detector import GapGuardDetector


class GapGuardSubscriber:

    def __init__(
        self,
        host="127.0.0.1",
        port=1884,
        telemetry_topic="gapguard/adsb/telemetry",
        alert_topic="gapguard/alerts",
    ):

        self.host = host
        self.port = port

        self.telemetry_topic = telemetry_topic
        self.alert_topic = alert_topic

        self.detector = GapGuardDetector()

        # IMPORTANT:
        # final_iot_system.py expects subscriber.results
        self.results = []

        self.telemetry_received = 0
        self.alerts_generated = 0
        self.unverifiable_count = 0
        self.suppressed_count = 0

        self.project_root = Path(
            __file__
        ).resolve().parents[2]

        self.results_dir = (
            self.project_root
            / "results"
            / "system"
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # AMQTT currently works with MQTT 3.1.1
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(
        self,
        client,
        userdata,
        flags,
        reason_code,
        properties=None,
    ):

        print(
            f"[MQTT] Connected "
            f"(reason={reason_code})"
        )

        client.subscribe(
            self.telemetry_topic,
            qos=0,
        )

        print(
            f"[MQTT] Subscribed: "
            f"{self.telemetry_topic}"
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

            # Run the streaming GapGuard detector.
            result = self.detector.process(
                payload
            )

            # Store every processed result.
            self.results.append(result)

            self.telemetry_received += 1

            decision = result.get(
                "decision",
                "SUPPRESS",
            )

            # ---------------------------------------------------------
            # Decision handling
            # ---------------------------------------------------------

            if decision == "ALERT":

                self.alerts_generated += 1

                client.publish(
                    self.alert_topic,
                    json.dumps(result),
                    qos=0,
                )

            elif decision == "UNVERIFIABLE":

                self.unverifiable_count += 1

            else:

                self.suppressed_count += 1

            # ---------------------------------------------------------
            # Display detector result
            # ---------------------------------------------------------

            timestamp = result.get(
                "timestamp",
                payload.get(
                    "t_elapsed_sec",
                    0,
                ),
            )

            # Detector returns "gap_seconds".
            gap = result.get(
                "gap_seconds",
                payload.get(
                    "dt",
                    0,
                ),
            )

            severity = result.get(
                "severity",
                0,
            )

            # Detector returns "evidence".
            # Convert to uppercase for a cleaner terminal display.
            evidence = result.get(
                "evidence",
                "unknown",
            ).upper()

            print(
                f"[DETECT] "
                f"t={float(timestamp):.2f} "
                f"| gap={float(gap):.2f}s "
                f"| severity={float(severity):.3f} "
                f"| evidence={evidence} "
                f"| decision={decision}"
            )

        except Exception as exc:

            print(
                f"[MQTT] Message processing error: "
                f"{exc}"
            )

    def run(self):

        self.client.connect(
            self.host,
            self.port,
            60,
        )

        self.client.loop_start()

        time.sleep(0.5)

    def stop(self):

        self.client.loop_stop()
        self.client.disconnect()

        self.save_results()

    def save_results(self):

        output = {
            "telemetry_received":
                self.telemetry_received,

            "alerts":
                self.alerts_generated,

            "unverifiable":
                self.unverifiable_count,

            "suppressed":
                self.suppressed_count,

            "results":
                self.results,
        }

        output_path = (
            self.results_dir
            / "mqtt_stream_results.json"
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

        print(
            f"[MQTT] Saved: {output_path}"
        )
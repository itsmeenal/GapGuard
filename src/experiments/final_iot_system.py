"""
GapGuard Stage 4.9
Complete IoT/MQTT streaming demonstration.

ONE COMMAND runs:

    AMQTT broker
        ↓
    MQTT publisher
        ↓
    MQTT subscriber
        ↓
    GapGuard detector
        ↓
    alert generation
        ↓
    result validation
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from amqtt.broker import Broker

from src.mqtt.publisher import main as publisher_main
from src.mqtt.subscriber import GapGuardSubscriber


BROKER_CONFIG = {
    "listeners": {
        "default": {
            "type": "tcp",
            "bind": "127.0.0.1:1884",
        }
    },
    "sys_interval": 10,
    "topic-check": {
        "enabled": False,
    },
}


async def start_broker():

    broker = Broker(
        BROKER_CONFIG
    )

    await broker.start()

    return broker


def run_publisher():

    # Give subscriber time to connect.
    time.sleep(1.5)

    publisher_main()


async def main():

    print("=" * 80)
    print("STAGE 4.9 — GAPGUARD COMPLETE IoT SYSTEM")
    print("=" * 80)
    print()

    broker = await start_broker()

    print("[1/4] AMQTT broker started")
    print("[2/4] Starting GapGuard subscriber")

    subscriber = GapGuardSubscriber()

    subscriber.run()

    print("[3/4] Starting ADS-B MQTT publisher")
    print()

    publisher_thread = threading.Thread(
        target=run_publisher,
        daemon=True,
    )

    publisher_thread.start()

    # Wait until publisher finishes.
    while publisher_thread.is_alive():
        await asyncio.sleep(0.25)

    # Give final MQTT messages time to arrive.
    await asyncio.sleep(2)

    subscriber.client.loop_stop()
    subscriber.client.disconnect()

    result_path = (
        subscriber.save_results()
    )

    alerts = [
        x
        for x in subscriber.results
        if x["decision"] == "ALERT"
    ]

    unverifiable = [
        x
        for x in subscriber.results
        if x["decision"] == "UNVERIFIABLE"
    ]

    suppressed = [
        x
        for x in subscriber.results
        if x["decision"] == "SUPPRESS"
    ]

    summary = {
        "telemetry_messages_received": len(
            subscriber.results
        ),
        "alerts": len(alerts),
        "unverifiable": len(
            unverifiable
        ),
        "suppressed": len(
            suppressed
        ),
        "mqtt_broker": "AMQTT",
        "mqtt_host": "127.0.0.1",
        "mqtt_port": 1884,
        "telemetry_topic":
            "gapguard/adsb/telemetry",
        "alert_topic":
            "gapguard/alerts",
        "detector":
            "delta-t-aware GapGuard",
        "calibration":
            "frozen clean validation calibration",
    }

    output_dir = Path(
        "results/system"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir /
        "final_iot_system_result.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    await broker.shutdown()

    print()
    print("=" * 80)
    print("FINAL IoT SYSTEM SUMMARY")
    print("=" * 80)

    print(
        f"Telemetry received : "
        f"{summary['telemetry_messages_received']}"
    )

    print(
        f"Alerts             : "
        f"{summary['alerts']}"
    )

    print(
        f"Unverifiable       : "
        f"{summary['unverifiable']}"
    )

    print(
        f"Suppressed         : "
        f"{summary['suppressed']}"
    )

    print()
    print(
        f"Saved: {result_path}"
    )

    print(
        f"Saved: {summary_path}"
    )

    if summary[
        "telemetry_messages_received"
    ] > 0:

        print()
        print(
            "STAGE 4.9 COMPLETE — "
            "GAPGUARD IoT PIPELINE RUNNING SUCCESSFULLY"
        )

    else:

        print()
        print(
            "STAGE 4.9 FAILED — "
            "NO TELEMETRY RECEIVED"
        )


if __name__ == "__main__":

    asyncio.run(main())
import asyncio
from amqtt.broker import Broker


BROKER_CONFIG = {
    "listeners": {
        "default": {
            "type": "tcp",
            "bind": "127.0.0.1:1884",
        }
    },
    "sys_interval": 10,
    "topic-check": {
        "enabled": False
    },
}


async def start_broker():
    broker = Broker(BROKER_CONFIG)
    await broker.start()
    return broker


async def run_broker_forever():
    broker = await start_broker()

    print("=" * 80)
    print("GAPGUARD MQTT BROKER")
    print("=" * 80)
    print("Broker : AMQTT")
    print("Host   : 127.0.0.1")
    print("Port   : 1884")
    print("Status : RUNNING")
    print()

    try:
        await asyncio.Event().wait()
    finally:
        await broker.shutdown()


if __name__ == "__main__":
    asyncio.run(run_broker_forever())
from __future__ import annotations

import json
import os
from pathlib import Path

from confluent_kafka import Consumer, KafkaError


OUTPUT_PATH = Path("outputs/day12/consumed_events.jsonl")


def main() -> None:
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092",
    )

    topic = os.getenv(
        "KAFKA_TOPIC",
        "trafficvision.events",
    )

    group_id = os.getenv(
        "KAFKA_GROUP_ID",
        "trafficvision-backend",
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,

            # New group starts from oldest available message.
            "auto.offset.reset": "earliest",

            # Commit only after our processing succeeds.
            "enable.auto.commit": False,
        }
    )

    consumer.subscribe([topic])

    print(f"[INFO] Bootstrap servers: {bootstrap_servers}")
    print(f"[INFO] Topic: {topic}")
    print(f"[INFO] Consumer group: {group_id}")
    print("[INFO] Waiting for TrafficVision events...")

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue

                print(f"[ERROR] Kafka error: {msg.error()}")
                continue

            raw_value = msg.value().decode("utf-8")

            try:
                event = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                print(
                    f"[ERROR] Invalid JSON "
                    f"partition={msg.partition()} "
                    f"offset={msg.offset()}: {exc}"
                )
                continue

            print()
            print(
                "[EVENT] "
                f"partition={msg.partition()} "
                f"offset={msg.offset()} "
                f"key={msg.key().decode('utf-8') if msg.key() else None}"
            )

            print(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            with OUTPUT_PATH.open(
                "a",
                encoding="utf-8",
            ) as file:
                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Commit AFTER successful local processing.
            consumer.commit(
                message=msg,
                asynchronous=False,
            )

    except KeyboardInterrupt:
        print("\n[INFO] Consumer interrupted")

    finally:
        consumer.close()
        print("[INFO] Consumer closed")


if __name__ == "__main__":
    main()

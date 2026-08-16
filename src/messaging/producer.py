from __future__ import annotations

import argparse
import json
import os
import socket
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrafficVision Kafka event producer"
    )

    parser.add_argument(
        "--camera-id",
        default="demo-cam-00",
        help="Logical camera identifier",
    )

    parser.add_argument(
        "--source-id",
        type=int,
        default=0,
        help="DeepStream source_id",
    )

    parser.add_argument(
        "--track-id",
        type=int,
        required=True,
        help="Tracker object_id / track_id",
    )

    parser.add_argument(
        "--vehicle-class",
        default="car",
        help="Detected vehicle class",
    )

    return parser.parse_args()


def delivery_report(err, msg) -> None:
    if err is not None:
        print(f"[ERROR] Delivery failed: {err}")
        return

    print(
        "[DELIVERED] "
        f"topic={msg.topic()} "
        f"partition={msg.partition()} "
        f"offset={msg.offset()}"
    )


def main() -> None:
    args = parse_args()

    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "localhost:9092",
    )

    topic = os.getenv(
        "KAFKA_TOPIC",
        "trafficvision.events",
    )

    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "client.id": socket.gethostname(),

            # Production-oriented delivery behaviour.
            "enable.idempotence": True,

            # Do not wait forever during this demo.
            "message.timeout.ms": 10000,
        }
    )

    event = {
        "schema_version": 1,
        "event_id": str(uuid.uuid4()),
        "event_type": "vehicle_observation",
        "occurred_at": datetime.now(timezone.utc).isoformat(),

        "source": {
            "camera_id": args.camera_id,
            "source_id": args.source_id,
        },

        "vehicle": {
            "track_id": args.track_id,
            "class": args.vehicle_class,
        },

        "anpr": {
            "plate_text": None,
            "confidence": None,
        },

        "speed": {
            "value_kmh": None,
            "validated": False,
        },

        "violation": {
            "type": None,
            "validated": False,
        },

        "pipeline": {
            "detector": "YOLO11n",
            "tracker": "NvDCF",
            "runtime": "DeepStream 7.1 / TensorRT FP16",
        },
    }

    payload = json.dumps(
        event,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    # Events belonging to the same camera + track use the same key.
    key = f"{args.camera_id}:{args.track_id}"

    print(f"[INFO] Bootstrap servers: {bootstrap_servers}")
    print(f"[INFO] Topic: {topic}")
    print(f"[INFO] Key: {key}")
    print(f"[INFO] Event:\n{json.dumps(event, indent=2)}")

    producer.produce(
        topic=topic,
        key=key.encode("utf-8"),
        value=payload.encode("utf-8"),
        callback=delivery_report,
    )

    remaining = producer.flush(10)

    if remaining != 0:
        raise RuntimeError(
            f"{remaining} Kafka message(s) were not delivered"
        )


if __name__ == "__main__":
    main()

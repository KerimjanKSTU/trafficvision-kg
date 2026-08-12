from __future__ import annotations

import argparse
import statistics
import time

import cv2
import torch
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    return parser.parse_args()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()

    print("Model:", args.model)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(args.model, task="detect")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    ok, warmup_frame = cap.read()

    if not ok:
        raise RuntimeError("Cannot read first video frame")

    print(f"Warm-up iterations: {args.warmup}")

    for _ in range(args.warmup):
        model.predict(
            warmup_frame,
            imgsz=args.imgsz,
            conf=0.10,
            iou=0.70,
            classes=VEHICLE_CLASSES,
            device=0,
            verbose=False,
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    latencies_ms = []

    while len(latencies_ms) < args.frames:
        ok, frame = cap.read()

        if not ok:
            break

        cuda_sync()
        start = time.perf_counter()

        model.predict(
            frame,
            imgsz=args.imgsz,
            conf=0.10,
            iou=0.70,
            classes=VEHICLE_CLASSES,
            device=0,
            verbose=False,
        )

        cuda_sync()

        latencies_ms.append(
            (time.perf_counter() - start) * 1000.0
        )

    cap.release()

    mean_ms = statistics.mean(latencies_ms)
    median_ms = statistics.median(latencies_ms)

    print()
    print("========== RESULT ==========")
    print(f"Frames:         {len(latencies_ms)}")
    print(f"Mean latency:   {mean_ms:.2f} ms")
    print(f"Median latency: {median_ms:.2f} ms")
    print(f"Min latency:    {min(latencies_ms):.2f} ms")
    print(f"Max latency:    {max(latencies_ms):.2f} ms")
    print(f"FPS:            {1000.0 / mean_ms:.2f}")


if __name__ == "__main__":
    main()

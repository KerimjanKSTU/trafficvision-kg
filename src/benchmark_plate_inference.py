from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark license plate detector inference."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to plate detector model",
    )

    parser.add_argument(
        "--images",
        default="data/calibration_plate_int8/images/val",
        help="Directory containing vehicle crops",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    return parser.parse_args()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()

    paths = sorted(
        Path(args.images).glob("*.jpg")
    )

    if not paths:
        raise RuntimeError(
            f"No JPG images found in {args.images}"
        )

    sample_count = min(
        args.frames,
        len(paths),
    )

    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(
                i * (len(paths) - 1)
                / (sample_count - 1)
            )
            for i in range(sample_count)
        ]

    selected_paths = [
        paths[index]
        for index in indices
    ]

    images = []

    for path in selected_paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        images.append(image)

    if not images:
        raise RuntimeError(
            "Could not load benchmark images"
        )

    print("Model:", args.model)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Images:", len(images))
    print("Image size:", args.imgsz)
    print("Confidence:", args.conf)

    model = YOLO(
        args.model,
        task="detect",
    )

    print(
        "Warm-up iterations:",
        args.warmup,
    )

    warmup_image = images[0]

    for _ in range(args.warmup):
        model.predict(
            warmup_image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=0,
            verbose=False,
        )

    latencies_ms = []

    total_detections = 0
    images_with_detection = 0
    confidence_values = []

    for image in images:
        cuda_sync()

        start = time.perf_counter()

        results = model.predict(
            image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=0,
            verbose=False,
        )

        cuda_sync()

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

        result = results[0]

        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):
            detections = len(
                result.boxes
            )

            total_detections += detections
            images_with_detection += 1

            confidence_values.extend(
                result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )

    mean_ms = statistics.mean(
        latencies_ms
    )

    median_ms = statistics.median(
        latencies_ms
    )

    mean_conf = (
        statistics.mean(
            confidence_values
        )
        if confidence_values
        else 0.0
    )

    print()
    print("========== RESULT ==========")
    print(
        f"Images:                "
        f"{len(latencies_ms)}"
    )
    print(
        f"Mean latency:          "
        f"{mean_ms:.2f} ms"
    )
    print(
        f"Median latency:        "
        f"{median_ms:.2f} ms"
    )
    print(
        f"Min latency:           "
        f"{min(latencies_ms):.2f} ms"
    )
    print(
        f"Max latency:           "
        f"{max(latencies_ms):.2f} ms"
    )
    print(
        f"FPS:                   "
        f"{1000.0 / mean_ms:.2f}"
    )

    print()
    print("====== DETECTIONS ======")
    print(
        f"Images with detection: "
        f"{images_with_detection}"
    )
    print(
        f"Total detections:       "
        f"{total_detections}"
    )
    print(
        f"Mean confidence:        "
        f"{mean_conf:.4f}"
    )


if __name__ == "__main__":
    main()

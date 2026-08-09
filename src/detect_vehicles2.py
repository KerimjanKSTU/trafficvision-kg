from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# COCO class IDs:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Day 2: детекция транспорта с анализом "
            "confidence, NMS и статистики."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Путь к входному видео.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/vehicles_detected_day2.mp4"),
        help="Путь для сохранения размеченного видео.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Название или путь к весам YOLO.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Минимальный confidence threshold.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold для NMS.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер изображения для YOLO.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(
            f"--conf должен быть от 0 до 1. Получено: {args.conf}"
        )

    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(
            f"--iou должен быть от 0 до 1. Получено: {args.iou}"
        )

    if args.imgsz <= 0:
        raise ValueError(
            f"--imgsz должен быть больше 0. Получено: {args.imgsz}"
        )

    if args.source.resolve() == args.output.resolve():
        raise ValueError(
            "Входное и выходное видео не могут иметь одинаковый путь."
        )


def validate_video(
    source: Path,
) -> tuple[float, int, int, int]:

    if not source.exists():
        raise FileNotFoundError(
            f"Видео не найдено: {source}"
        )

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV не смог открыть видео: {source}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    capture.release()

    if fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "Не удалось определить размер видео."
        )

    return fps, width, height, total_frames


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Не удалось создать видео: {output_path}"
        )

    return writer


def get_class_name(
    class_names: dict | list,
    class_id: int,
) -> str:

    if isinstance(class_names, dict):
        return str(
            class_names.get(
                class_id,
                f"class_{class_id}",
            )
        )

    if 0 <= class_id < len(class_names):
        return str(class_names[class_id])

    return f"class_{class_id}"


def main() -> None:
    args = parse_args()

    validate_args(args)

    fps, width, height, total_frames = validate_video(
        args.source
    )

    if torch.cuda.is_available():
        device: int | str = 0
        device_name = "CUDA GPU"
    else:
        device = "cpu"
        device_name = "CPU"

    print("=" * 70)
    print("DAY 2 — YOLO VEHICLE DETECTION")
    print("=" * 70)

    print(f"Source:        {args.source}")
    print(f"Output:        {args.output}")
    print(f"Model:         {args.model}")
    print(f"Device:        {device_name}")

    print("-" * 70)

    print(f"Resolution:    {width}x{height}")
    print(f"Video FPS:     {fps:.2f}")
    print(f"Frames:        {total_frames}")

    print("-" * 70)

    print(f"Confidence:    {args.conf}")
    print(f"NMS IoU:       {args.iou}")
    print(f"Image size:    {args.imgsz}")
    print(f"Vehicle IDs:   {VEHICLE_CLASS_IDS}")

    print("=" * 70)

    model = YOLO(args.model)

    writer = create_video_writer(
        args.output,
        fps,
        width,
        height,
    )

    results = model.predict(
        source=str(args.source),
        stream=True,
        classes=VEHICLE_CLASS_IDS,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )

    processed_frames = 0
    total_detections = 0
    frames_with_vehicles = 0

    class_counts: Counter[str] = Counter()

    class_confidence_sum: defaultdict[str, float] = (
        defaultdict(float)
    )

    total_confidence = 0.0

    min_confidence: float | None = None
    max_confidence: float | None = None

    start_time = time.perf_counter()

    try:
        for result in results:
            processed_frames += 1

            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:

                detections_in_frame = len(boxes)

                total_detections += detections_in_frame
                frames_with_vehicles += 1

                class_ids = (
                    boxes.cls
                    .detach()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    boxes.conf
                    .detach()
                    .cpu()
                    .tolist()
                )

                for class_id_raw, confidence_raw in zip(
                    class_ids,
                    confidences,
                ):
                    class_id = int(class_id_raw)
                    confidence = float(confidence_raw)

                    class_name = get_class_name(
                        result.names,
                        class_id,
                    )

                    class_counts[class_name] += 1

                    class_confidence_sum[
                        class_name
                    ] += confidence

                    total_confidence += confidence

                    if min_confidence is None:
                        min_confidence = confidence
                    else:
                        min_confidence = min(
                            min_confidence,
                            confidence,
                        )

                    if max_confidence is None:
                        max_confidence = confidence
                    else:
                        max_confidence = max(
                            max_confidence,
                            confidence,
                        )

            annotated_frame = result.plot()

            writer.write(annotated_frame)

            if processed_frames % 30 == 0:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                pipeline_fps = (
                    processed_frames / elapsed
                    if elapsed > 0
                    else 0.0
                )

                average_confidence = (
                    total_confidence
                    / total_detections
                    if total_detections > 0
                    else 0.0
                )

                print(
                    f"Frames: "
                    f"{processed_frames}/"
                    f"{total_frames or '?'} | "
                    f"Detections: "
                    f"{total_detections} | "
                    f"Avg conf: "
                    f"{average_confidence:.3f} | "
                    f"FPS: "
                    f"{pipeline_fps:.2f}"
                )

    finally:
        writer.release()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    pipeline_fps = (
        processed_frames / elapsed
        if elapsed > 0
        else 0.0
    )

    average_confidence = (
        total_confidence / total_detections
        if total_detections > 0
        else 0.0
    )

    detections_per_frame = (
        total_detections / processed_frames
        if processed_frames > 0
        else 0.0
    )

    vehicle_frame_ratio = (
        frames_with_vehicles / processed_frames
        if processed_frames > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("DAY 2 RESULTS")
    print("=" * 70)

    print(
        f"Processed frames:       "
        f"{processed_frames}"
    )

    print(
        f"Detection occurrences:  "
        f"{total_detections}"
    )

    print(
        f"Frames with vehicles:   "
        f"{frames_with_vehicles}"
    )

    print(
        f"Vehicle frame ratio:    "
        f"{vehicle_frame_ratio:.2%}"
    )

    print(
        f"Detections per frame:   "
        f"{detections_per_frame:.2f}"
    )

    if total_detections > 0:

        print(
            f"Average confidence:     "
            f"{average_confidence:.4f}"
        )

        print(
            f"Minimum confidence:     "
            f"{min_confidence:.4f}"
        )

        print(
            f"Maximum confidence:     "
            f"{max_confidence:.4f}"
        )

    print()
    print("DETECTIONS BY CLASS")
    print("-" * 70)

    if class_counts:

        for class_name, count in sorted(
            class_counts.items()
        ):

            class_average_confidence = (
                class_confidence_sum[class_name]
                / count
            )

            print(
                f"{class_name:<15} "
                f"{count:>8} | "
                f"avg conf: "
                f"{class_average_confidence:.4f}"
            )

    else:
        print("No vehicles detected.")

    print()
    print("-" * 70)

    print(
        f"Elapsed time:           "
        f"{elapsed:.2f} s"
    )

    print(
        f"Average pipeline FPS:   "
        f"{pipeline_fps:.2f}"
    )

    print(
        f"Output video:           "
        f"{args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
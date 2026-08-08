from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# Идентификаторы транспортных средств в датасете COCO.
VEHICLE_CLASS_IDS = [2, 3, 5, 7]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Детекция транспорта на видео с помощью YOLO."
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
        default=Path("data/output/vehicles_detected.mp4"),
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
        help="Минимальный confidence для сохранения детекции.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU-порог для NMS.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер изображения, подаваемого модели.",
    )

    return parser.parse_args()


def validate_video(source: Path) -> tuple[float, int, int, int]:
    if not source.exists():
        raise FileNotFoundError(f"Видео не найдено: {source}")

    capture = cv2.VideoCapture(str(source))

    if not capture.isOpened():
        raise RuntimeError(f"OpenCV не смог открыть видео: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.release()

    if fps <= 0:
        fps = 25.0

    if width <= 0 or height <= 0:
        raise RuntimeError("Не удалось определить размер видео.")

    return fps, width, height, total_frames


def create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Не удалось создать выходное видео: {output_path}"
        )

    return writer


def main() -> None:
    args = parse_args()

    fps, width, height, total_frames = validate_video(args.source)

    device: int | str = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Детекция транспорта")
    print(f"Входное видео: {args.source}")
    print(f"Выходное видео: {args.output}")
    print(f"Модель: {args.model}")
    print(f"Устройство: {device}")
    print(f"Размер видео: {width}x{height}")
    print(f"Исходный FPS: {fps:.2f}")
    print(f"Количество кадров: {total_frames}")
    print(f"Confidence: {args.conf}")
    print(f"IoU NMS: {args.iou}")
    print(f"Image size: {args.imgsz}")
    print("=" * 60)

    model = YOLO(args.model)

    writer = create_video_writer(
        output_path=args.output,
        fps=fps,
        width=width,
        height=height,
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

    start_time = time.perf_counter()

    processed_frames = 0
    total_detections = 0
    frames_with_vehicles = 0

    try:
        for result in results:
            processed_frames += 1

            detections_in_frame = (
                len(result.boxes)
                if result.boxes is not None
                else 0
            )

            total_detections += detections_in_frame

            if detections_in_frame > 0:
                frames_with_vehicles += 1

            # result.plot() возвращает кадр с нарисованными рамками.
            annotated_frame = result.plot()

            writer.write(annotated_frame)

            if processed_frames % 30 == 0:
                elapsed = time.perf_counter() - start_time
                pipeline_fps = processed_frames / elapsed

                print(
                    f"Обработано кадров: {processed_frames}/"
                    f"{total_frames or '?'} | "
                    f"Текущий pipeline FPS: {pipeline_fps:.2f}"
                )

    finally:
        writer.release()

    elapsed = time.perf_counter() - start_time
    pipeline_fps = processed_frames / elapsed if elapsed > 0 else 0.0

    print("\nОбработка завершена")
    print(f"Обработано кадров: {processed_frames}")
    print(f"Всего детекций: {total_detections}")
    print(f"Кадров с транспортом: {frames_with_vehicles}")
    print(f"Полное время: {elapsed:.2f} секунд")
    print(f"Средний pipeline FPS: {pipeline_fps:.2f}")
    print(f"Результат: {args.output}")


if __name__ == "__main__":
    main()
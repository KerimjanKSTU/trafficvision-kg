from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare representative vehicle crops "
            "for plate-detector INT8 calibration."
        )
    )

    parser.add_argument(
        "--model",
        default="yolo11n_fp16.engine",
        help="Vehicle detector",
    )

    parser.add_argument(
        "--images",
        default="data/calibration_int8/images/val",
        help="Representative full-frame images",
    )

    parser.add_argument(
        "--output",
        default="data/calibration_plate_int8/images/val",
        help="Final plate-model calibration images",
    )

    parser.add_argument(
        "--candidates",
        default="data/calibration_plate_int8/candidates",
        help="Temporary directory for all vehicle crops",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum final calibration images",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--min-width",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=30,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.images)
    candidate_dir = Path(args.candidates)
    output_dir = Path(args.output)

    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    candidate_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(image_dir.glob("*.jpg"))

    if not paths:
        raise RuntimeError(
            f"No calibration frames found in {image_dir}"
        )

    print("[INFO] Loading vehicle model:", args.model)
    model = YOLO(args.model, task="detect")

    candidate_paths = []
    crop_number = 0

    for frame_number, path in enumerate(paths):
        frame = cv2.imread(str(path))

        if frame is None:
            print("[WARN] Cannot read:", path)
            continue

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.70,
            classes=VEHICLE_CLASSES,
            verbose=False,
        )

        if not results:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        h, w = frame.shape[:2]

        for box in result.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]

            crop_h, crop_w = crop.shape[:2]

            if (
                crop_w < args.min_width
                or crop_h < args.min_height
            ):
                continue

            crop_path = candidate_dir / (
                f"frame_{frame_number:04d}_"
                f"vehicle_{crop_number:06d}.jpg"
            )

            if cv2.imwrite(str(crop_path), crop):
                candidate_paths.append(crop_path)
                crop_number += 1

    print("[INFO] Vehicle crops found:", len(candidate_paths))

    if not candidate_paths:
        raise RuntimeError(
            "Vehicle detector produced no calibration crops"
        )

    final_count = min(
        args.max_images,
        len(candidate_paths),
    )

    if final_count == 1:
        selected_indices = [0]
    else:
        selected_indices = [
            round(
                i * (len(candidate_paths) - 1)
                / (final_count - 1)
            )
            for i in range(final_count)
        ]

    for output_index, candidate_index in enumerate(selected_indices):
        source = candidate_paths[candidate_index]

        destination = (
            output_dir /
            f"vehicle_{output_index:06d}.jpg"
        )

        shutil.copy2(
            source,
            destination,
        )

    print("[DONE] Final calibration images:", final_count)
    print("[DONE] Output:", output_dir)


if __name__ == "__main__":
    main()

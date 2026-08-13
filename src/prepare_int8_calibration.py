from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare representative frames for TensorRT INT8 calibration."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Input traffic video",
    )

    parser.add_argument(
        "--output",
        default="data/calibration_int8/images/val",
        help="Output directory for calibration images",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Number of frames sampled uniformly from the video",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source)
    output_dir = Path(args.output)

    if args.count < 1:
        raise ValueError("--count must be >= 1")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {source}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        raise RuntimeError(
            "Video reports zero frames"
        )

    sample_count = min(
        args.count,
        total_frames,
    )

    if sample_count == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            round(
                i * (total_frames - 1)
                / (sample_count - 1)
            )
            for i in range(sample_count)
        ]

    saved = 0

    print("Source:", source)
    print("Total video frames:", total_frames)
    print("Requested calibration frames:", sample_count)

    for frame_idx in frame_indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx,
        )

        ok, frame = cap.read()

        if not ok:
            print(
                f"[WARN] Cannot read frame {frame_idx}"
            )
            continue

        output_path = (
            output_dir /
            f"frame_{frame_idx:06d}.jpg"
        )

        ok = cv2.imwrite(
            str(output_path),
            frame,
        )

        if not ok:
            print(
                f"[WARN] Cannot save {output_path}"
            )
            continue

        saved += 1

    cap.release()

    print()
    print("Saved calibration frames:", saved)
    print("Output:", output_dir)


if __name__ == "__main__":
    main()

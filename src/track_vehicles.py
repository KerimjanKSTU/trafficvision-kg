import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# COCO classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle tracking with YOLO + ByteTrack"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        default="outputs/day3_bytetrack.mp4",
        help="Path to output video",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO model",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show tracking window",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Opening video: {args.source}")
    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 1:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps:.2f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    # История движения каждого track_id.
    track_history = defaultdict(list)

    # Все ID, которые когда-либо встретились.
    # Пока используем только для отладки.
    seen_ids = set()

    frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # ВАЖНО:
        # persist=True сохраняет tracker state
        # между последовательными кадрами.
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        result = results[0]

        annotated_frame = frame.copy()

        active_tracks = 0

        if result.boxes is not None and result.boxes.is_track:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, class_id, confidence in zip(
                boxes,
                track_ids,
                class_ids,
                confidences,
            ):

                x1, y1, x2, y2 = map(int, box)

                active_tracks += 1
                seen_ids.add(track_id)

                class_name = model.names[class_id]

                # Центр bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Сохраняем историю движения
                track_history[track_id].append(
                    (center_x, center_y)
                )

                # Храним только последние 30 точек
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Bounding box
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Рисуем траекторию
                points = np.array(
                    track_history[track_id],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))

                if len(points) > 1:
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(255, 255, 0),
                        thickness=2,
                    )

        # Статистика
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Active tracks: {active_tracks}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Unique IDs: {len(seen_ids)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(annotated_frame)

        if args.show:
            cv2.imshow(
                "TrafficVision KG - ByteTrack",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print("[DONE] Tracking finished")
    print(f"[DONE] Output: {output_path}")
    print(f"[DONE] Unique track IDs: {len(seen_ids)}")


if __name__ == "__main__":
    main()
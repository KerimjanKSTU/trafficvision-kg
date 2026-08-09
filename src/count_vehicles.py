import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# COCO vehicle classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vehicle tracking and counting with YOLO + ByteTrack"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        default="outputs/day4_bytetrack_counting.mp4",
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
        "--count-line-y",
        type=int,
        default=500,
        help="Y coordinate of counting line",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show tracking window",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1. Загружаем YOLO
    # ---------------------------------------------------------

    print(f"[INFO] Loading model: {args.model}")

    model = YOLO(args.model)

    # ---------------------------------------------------------
    # 2. Открываем видео
    # ---------------------------------------------------------

    print(f"[INFO] Opening video: {args.source}")

    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.source}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 1:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"[INFO] Resolution: {width}x{height}"
    )

    print(
        f"[INFO] FPS: {fps:.2f}"
    )

    # ---------------------------------------------------------
    # 3. Проверяем counting line
    # ---------------------------------------------------------

    count_line_y = args.count_line_y

    if count_line_y >= height:
        old_value = count_line_y

        count_line_y = int(
            height * 0.70
        )

        print(
            f"[WARNING] count-line-y={old_value} "
            f"is outside video height={height}"
        )

        print(
            f"[WARNING] Using count-line-y="
            f"{count_line_y}"
        )

    print(
        f"[INFO] Counting line Y: "
        f"{count_line_y}"
    )

    # ---------------------------------------------------------
    # 4. Создаём output
    # ---------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    # ---------------------------------------------------------
    # 5. Хранилища для треков
    # ---------------------------------------------------------

    # История движения каждого track_id
    track_history = defaultdict(list)

    # Все ID транспортных средств,
    # которые встретились за видео
    seen_ids = set()

    # ID, которые уже были посчитаны по направлениям
    counted_up_ids = set()
    counted_down_ids = set()

    frame_number = 0
    up_count = 0
    down_count = 0

    # ---------------------------------------------------------
    # 6. Основной цикл
    # ---------------------------------------------------------

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -----------------------------------------------------
        # YOLO + ByteTrack
        # -----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",

            # КРИТИЧЕСКИ ВАЖНО:
            # трекаем только транспорт
            classes=VEHICLE_CLASSES,

            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        result = results[0]

        annotated_frame = frame.copy()

        active_tracks = 0

        # -----------------------------------------------------
        # Counting line
        # -----------------------------------------------------

        cv2.line(
            annotated_frame,
            (0, count_line_y),
            (width, count_line_y),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            annotated_frame,
            "COUNT LINE",
            (20, max(count_line_y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # -----------------------------------------------------
        # Обрабатываем tracks
        # -----------------------------------------------------

        if (
            result.boxes is not None
            and result.boxes.is_track
            and result.boxes.id is not None
        ):

            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            class_ids = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            for (
                box,
                track_id,
                class_id,
                confidence,
            ) in zip(
                boxes,
                track_ids,
                class_ids,
                confidences,
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box,
                )

                active_tracks += 1

                seen_ids.add(track_id)

                class_name = (
                    model.names[class_id]
                )

                # -------------------------------------------------
                # Нижняя центральная точка bounding box
                #
                # В День 3 использовали обычный центр.
                # Для дорожной аналитики лучше нижний центр,
                # потому что он ближе к точке контакта машины
                # с дорогой.
                # -------------------------------------------------

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = y2

                # -------------------------------------------------
                # Предыдущая позиция
                # -------------------------------------------------

                history = (
                    track_history[track_id]
                )

                previous_y = None

                if len(history) > 0:
                    previous_y = (
                        history[-1][1]
                    )

                # -------------------------------------------------
                # Добавляем новую точку
                # -------------------------------------------------

                history.append(
                    (
                        center_x,
                        center_y,
                    )
                )

                # Оставляем последние 30 точек
                if len(history) > 30:
                    history.pop(0)

                # -------------------------------------------------
                # Проверяем пересечение линии
                # -------------------------------------------------

                if previous_y is not None:

                    # Движение сверху вниз
                    crossed_down = (
                        previous_y
                        < count_line_y
                        <= center_y
                    )

                    # Движение снизу вверх
                    crossed_up = (
                        previous_y
                        > count_line_y
                        >= center_y
                    )

                    if (
                        crossed_down
                        and track_id
                        not in counted_down_ids
                    ):
                        counted_down_ids.add(track_id)
                        down_count += 1

                        print(
                            f"[COUNT DOWN] "
                            f"{class_name} "
                            f"ID:{track_id} "
                            f"Down:{down_count}"
                        )

                    elif (
                        crossed_up
                        and track_id
                        not in counted_up_ids
                    ):
                        counted_up_ids.add(track_id)
                        up_count += 1

                        print(
                            f"[COUNT UP] "
                            f"{class_name} "
                            f"ID:{track_id} "
                            f"Up:{up_count}"
                        )

                # -------------------------------------------------
                # Bounding box
                # -------------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # Label
                # -------------------------------------------------

                label = (
                    f"{class_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (
                        x1,
                        max(y1 - 10, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # -------------------------------------------------
                # Нижняя центральная точка
                # -------------------------------------------------

                cv2.circle(
                    annotated_frame,
                    (
                        center_x,
                        center_y,
                    ),
                    4,
                    (0, 0, 255),
                    -1,
                )

                # -------------------------------------------------
                # Траектория
                # -------------------------------------------------

                points = np.array(
                    history,
                    dtype=np.int32,
                ).reshape(
                    (-1, 1, 2)
                )

                if len(points) > 1:
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(255, 255, 0),
                        thickness=2,
                    )

        # ---------------------------------------------------------
        # 7. Статистика
        # ---------------------------------------------------------

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
            f"Unique vehicle IDs: {len(seen_ids)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        total_count = up_count + down_count

        cv2.putText(
            annotated_frame,
            f"UP: {up_count}",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"DOWN: {down_count}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"TOTAL: {total_count}",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        # ---------------------------------------------------------
        # 8. СОХРАНЯЕМ КАЖДЫЙ КАДР
        # ---------------------------------------------------------

        writer.write(
            annotated_frame
        )

        # ---------------------------------------------------------
        # 9. Показываем окно только при --show
        # ---------------------------------------------------------

        if args.show:
            cv2.imshow(
                "TrafficVision KG - Day 4",
                annotated_frame,
            )

            if (
                cv2.waitKey(1)
                & 0xFF
                == ord("q")
            ):
                break

    # ---------------------------------------------------------
    # 10. Завершение
    # ---------------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print(
        "[DONE] Vehicle counting finished"
    )

    print(
        f"[DONE] Output: {output_path}"
    )

    print(
        f"[DONE] Unique vehicle IDs: "
        f"{len(seen_ids)}"
    )

    total_count = up_count + down_count

    print(
        f"[DONE] Vehicles UP: "
        f"{up_count}"
    )

    print(
        f"[DONE] Vehicles DOWN: "
        f"{down_count}"
    )

    print(
        f"[DONE] Total counted vehicles: "
        f"{total_count}"
    )


if __name__ == "__main__":
    main()
import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


# COCO vehicle classes:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]

OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MAX_PLATE_HISTORY = 10
MIN_OCR_CONFIDENCE = 0.30

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Vehicle tracking, counting and ANPR "
            "with YOLO + ByteTrack + EasyOCR"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        default="outputs/day5_anpr.mp4",
        help="Path to output video",
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO vehicle model",
    )

    parser.add_argument(
        "--plate-model",
        default="models/license_plate_detector.pt",
        help="YOLO license plate detector",
    )

    parser.add_argument(
        "--plate-conf",
        type=float,
        default=0.25,
        help="License plate confidence threshold",
    )

    parser.add_argument(
        "--plate-imgsz",
        type=int,
        default=960,
        help="License plate detector image size",
    )

    parser.add_argument(
        "--ocr-every",
        type=int,
        default=5,
        help="Run plate detection + OCR every N frames",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Vehicle confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="Vehicle IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Vehicle inference image size",
    )

    parser.add_argument(
        "--count-line-y",
        type=int,
        default=500,
        help="Y coordinate of counting line",
    )

    parser.add_argument(
        "--save-plates",
        action="store_true",
        help="Save detected license-plate crops to outputs/plates",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show tracking window",
    )

    return parser.parse_args()


def normalize_plate_text(text):
    """Uppercase OCR text and keep only Latin letters and digits."""
    if not text:
        return ""

    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def is_plausible_plate(text):
    """Soft baseline filter. Exact Kyrgyz plate masks can be added later."""
    if not text:
        return False

    if len(text) < 5 or len(text) > 12:
        return False

    return text.isalnum()


def preprocess_plate(image):
    """Prepare a plate crop for OCR without aggressive image processing."""
    if image is None or image.size == 0:
        return image

    _, width = image.shape[:2]

    # Small plate crops are difficult for OCR, so enlarge them first.
    if width < 160:
        scale = 160 / max(width, 1)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE is gentler than global equalization for uneven illumination.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    gray = clahe.apply(gray)

    return gray


def recognize_plate(ocr_reader, image):
    """Run EasyOCR and merge all detected text fragments left-to-right."""
    if image is None or image.size == 0:
        return "", 0.0

    ocr_results = ocr_reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=OCR_ALLOWLIST,
    )

    if not ocr_results:
        return "", 0.0

    # A plate may be returned as several text fragments.
    ocr_results = sorted(
        ocr_results,
        key=lambda item: min(point[0] for point in item[0]),
    )

    fragments = []
    weighted_confidence = 0.0
    total_characters = 0

    for _, text, confidence in ocr_results:
        normalized_fragment = normalize_plate_text(text)

        if not normalized_fragment:
            continue

        fragments.append(normalized_fragment)

        fragment_length = len(normalized_fragment)
        weighted_confidence += float(confidence) * fragment_length
        total_characters += fragment_length

    if not fragments or total_characters == 0:
        return "", 0.0

    text = "".join(fragments)
    confidence = weighted_confidence / total_characters

    return text, confidence


def update_plate_history(
    plate_history,
    best_plate_by_id,
    track_id,
    plate_text,
    ocr_confidence,
):
    """Add one OCR observation and choose the best plate by weighted voting."""
    if not is_plausible_plate(plate_text):
        return best_plate_by_id.get(track_id, "")

    history = plate_history[track_id]
    history.append((plate_text, float(ocr_confidence)))

    if len(history) > MAX_PLATE_HISTORY:
        history.pop(0)

    scores = defaultdict(float)
    counts = defaultdict(int)

    for text, confidence in history:
        scores[text] += confidence
        counts[text] += 1

    # Primary criterion: accumulated OCR confidence.
    # Secondary criterion: number of repeated observations.
    best_plate = max(
        scores,
        key=lambda text: (scores[text], counts[text]),
    )

    best_plate_by_id[track_id] = best_plate
    return best_plate


def main():
    args = parse_args()

    if args.ocr_every < 1:
        raise ValueError("--ocr-every must be >= 1")

    # ---------------------------------------------------------
    # 1. Models + OCR
    # ---------------------------------------------------------

    print(f"[INFO] Loading vehicle model: {args.model}")
    vehicle_model = YOLO(args.model)

    print(f"[INFO] Loading plate model: {args.plate_model}")
    plate_model = YOLO(args.plate_model)

    print("[INFO] Loading EasyOCR")
    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=False,
    )
    print("[INFO] EasyOCR loaded")

    # ---------------------------------------------------------
    # 2. Input video
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps:.2f}")

    # ---------------------------------------------------------
    # 3. Counting line
    # ---------------------------------------------------------

    count_line_y = args.count_line_y

    if count_line_y >= height or count_line_y < 0:
        old_value = count_line_y
        count_line_y = int(height * 0.70)

        print(
            f"[WARNING] count-line-y={old_value} "
            f"is outside video height={height}"
        )
        print(
            f"[WARNING] Using count-line-y={count_line_y}"
        )

    print(f"[INFO] Counting line Y: {count_line_y}")

    # ---------------------------------------------------------
    # 4. Output
    # ---------------------------------------------------------

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plate_output_dir = Path("outputs/plates")

    if args.save_plates:
        plate_output_dir.mkdir(
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
            f"Could not create output video: {output_path}"
        )

    # ---------------------------------------------------------
    # 5. Track + ANPR state
    # ---------------------------------------------------------

    track_history = defaultdict(list)
    seen_ids = set()

    counted_up_ids = set()
    counted_down_ids = set()

    # track_id -> [(plate_text, ocr_confidence), ...]
    plate_history = defaultdict(list)

    # track_id -> stable voted plate
    best_plate_by_id = {}

    frame_number = 0
    up_count = 0
    down_count = 0

    # ---------------------------------------------------------
    # 6. Main loop
    # ---------------------------------------------------------

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # -----------------------------------------------------
        # Vehicle detection + ByteTrack
        # -----------------------------------------------------

        results = vehicle_model.track(
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

        # Plate detection + OCR is intentionally throttled.
        run_ocr = frame_number % args.ocr_every == 0

        # -----------------------------------------------------
        # Process tracks
        # -----------------------------------------------------

        if (
            result.boxes is not None
            and result.boxes.is_track
            and result.boxes.id is not None
        ):
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

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
                x1, y1, x2, y2 = map(int, box)

                # Keep bbox inside the image.
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                active_tracks += 1
                seen_ids.add(track_id)

                class_name = vehicle_model.names[class_id]

                # -------------------------------------------------
                # Day 5: vehicle ROI -> plate detector -> OCR
                # -------------------------------------------------

                vehicle_crop = frame[y1:y2, x1:x2]

                if run_ocr and vehicle_crop.size > 0:
                    plate_results = plate_model.predict(
                        vehicle_crop,
                        conf=args.plate_conf,
                        imgsz=args.plate_imgsz,
                        verbose=False,
                    )

                    if plate_results:
                        plate_result = plate_results[0]

                        if (
                            plate_result.boxes is not None
                            and len(plate_result.boxes) > 0
                        ):
                            # Use the most confident plate detection.
                            plate_confidences = (
                                plate_result.boxes.conf
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            best_index = int(
                                np.argmax(plate_confidences)
                            )

                            plate_box = (
                                plate_result.boxes.xyxy[best_index]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            plate_detection_confidence = float(
                                plate_confidences[best_index]
                            )

                            px1, py1, px2, py2 = map(
                                int,
                                plate_box,
                            )

                            vehicle_h, vehicle_w = (
                                vehicle_crop.shape[:2]
                            )

                            px1 = max(0, px1)
                            py1 = max(0, py1)
                            px2 = min(vehicle_w, px2)
                            py2 = min(vehicle_h, py2)

                            if px2 > px1 and py2 > py1:
                                plate_crop = vehicle_crop[
                                    py1:py2,
                                    px1:px2,
                                ]

                                if plate_crop.size > 0:
                                    plate_h, plate_w = (
                                        plate_crop.shape[:2]
                                    )

                                    # Ignore extremely tiny detections.
                                    if plate_w >= 20 and plate_h >= 8:
                                        # Convert plate coordinates from
                                        # vehicle ROI back to full frame.
                                        global_px1 = x1 + px1
                                        global_py1 = y1 + py1
                                        global_px2 = x1 + px2
                                        global_py2 = y1 + py2

                                        cv2.rectangle(
                                            annotated_frame,
                                            (
                                                global_px1,
                                                global_py1,
                                            ),
                                            (
                                                global_px2,
                                                global_py2,
                                            ),
                                            (0, 255, 255),
                                            2,
                                        )

                                        if args.save_plates:
                                            plate_file = (
                                                plate_output_dir
                                                / (
                                                    f"frame_{frame_number:06d}_"
                                                    f"id_{track_id}.jpg"
                                                )
                                            )
                                            cv2.imwrite(
                                                str(plate_file),
                                                plate_crop,
                                            )

                                        processed_plate = (
                                            preprocess_plate(
                                                plate_crop
                                            )
                                        )

                                        (
                                            plate_text,
                                            ocr_confidence,
                                        ) = recognize_plate(
                                            ocr_reader,
                                            processed_plate,
                                        )

                                        plate_text = (
                                            normalize_plate_text(
                                                plate_text
                                            )
                                        )

                                       
                                        if (
                                            is_plausible_plate(plate_text)
                                            and ocr_confidence >= MIN_OCR_CONFIDENCE
                                        ):
                                            best_plate = (
                                                update_plate_history(
                                                    plate_history,
                                                    best_plate_by_id,
                                                    track_id,
                                                    plate_text,
                                                    ocr_confidence,
                                                )
                                            )

                                            print(
                                                f"[ANPR] "
                                                f"frame={frame_number} "
                                                f"ID:{track_id} "
                                                f"plate={plate_text} "
                                                f"ocr={ocr_confidence:.2f} "
                                                f"det={plate_detection_confidence:.2f} "
                                                f"best={best_plate}"
                                            )

                # -------------------------------------------------
                # Lower-center point for road analytics
                # -------------------------------------------------

                center_x = int((x1 + x2) / 2)
                center_y = y2

                history = track_history[track_id]
                previous_y = None

                if history:
                    previous_y = history[-1][1]

                history.append((center_x, center_y))

                if len(history) > 30:
                    history.pop(0)

                # -------------------------------------------------
                # Counting-line crossing
                # -------------------------------------------------

                if previous_y is not None:
                    crossed_down = (
                        previous_y
                        < count_line_y
                        <= center_y
                    )

                    crossed_up = (
                        previous_y
                        > count_line_y
                        >= center_y
                    )

                    if (
                        crossed_down
                        and track_id not in counted_down_ids
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
                        and track_id not in counted_up_ids
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
                # Vehicle bbox + stable ANPR label
                # -------------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                best_plate = best_plate_by_id.get(
                    track_id,
                    "",
                )

                if best_plate:
                    label = (
                        f"{class_name} "
                        f"ID:{track_id} "
                        f"{best_plate} "
                        f"{confidence:.2f}"
                    )
                else:
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

                # -------------------------------------------------
                # Lower-center point
                # -------------------------------------------------

                cv2.circle(
                    annotated_frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1,
                )

                # -------------------------------------------------
                # Trajectory
                # -------------------------------------------------

                points = np.array(
                    history,
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

        # ---------------------------------------------------------
        # 7. Statistics
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
        # 8. Save every frame
        # ---------------------------------------------------------

        writer.write(annotated_frame)

        # ---------------------------------------------------------
        # 9. Optional preview
        # ---------------------------------------------------------

        if args.show:
            cv2.imshow(
                "TrafficVision KG - Day 5 ANPR",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ---------------------------------------------------------
    # 10. Cleanup
    # ---------------------------------------------------------

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    total_count = up_count + down_count

    print()
    print("[DONE] Day 5 ANPR finished")
    print(f"[DONE] Output: {output_path}")
    print(f"[DONE] Unique vehicle IDs: {len(seen_ids)}")
    print(f"[DONE] Vehicles UP: {up_count}")
    print(f"[DONE] Vehicles DOWN: {down_count}")
    print(f"[DONE] Total counted vehicles: {total_count}")
    print(f"[DONE] Tracks with recognized plates: {len(best_plate_by_id)}")


if __name__ == "__main__":
    main()

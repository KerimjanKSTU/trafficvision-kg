from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
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


@dataclass(eq=False)
class PlateCandidate:
    """One plate crop candidate associated with a ByteTrack track_id."""

    score: float
    frame_idx: int
    crop: np.ndarray
    plate_conf: float
    sharpness: float
    width: int
    height: int
    plate_text: str = ""
    ocr_conf: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TrafficVision KG Day 7 Demo: YOLO + ByteTrack + vehicle counting + "
            "best-shot ANPR + experimental homography/speed"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to clean input video, e.g. data/input/intersection.mp4",
    )
    parser.add_argument(
        "--output",
        default="outputs/day7_demo.mp4",
        help="Path to annotated output video",
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

    # Vehicle detector / tracker parameters.
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
        help="Vehicle NMS IoU threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Vehicle detector inference image size",
    )

    # Plate detector / best-shot parameters.
    parser.add_argument(
        "--plate-conf",
        type=float,
        default=0.25,
        help="YOLO plate detector confidence threshold",
    )
    parser.add_argument(
        "--plate-imgsz",
        type=int,
        default=960,
        help="Plate detector inference image size",
    )
    parser.add_argument(
        "--ocr-every",
        type=int,
        default=5,
        help=(
            "Evaluate plate detection / best-shot candidates every N frames. "
            "Kept for compatibility with Day 5."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Keep this many best plate crops per track_id",
    )
    parser.add_argument(
        "--min-plate-width",
        type=int,
        default=35,
        help="Reject plate crops narrower than this many pixels",
    )
    parser.add_argument(
        "--min-plate-height",
        type=int,
        default=10,
        help="Reject plate crops shorter than this many pixels",
    )
    parser.add_argument(
        "--min-quality-plate-conf",
        type=float,
        default=0.40,
        help=(
            "Reject best-shot candidates below this plate confidence. "
            "This is a heuristic Day 6 threshold, not a calibrated production value."
        ),
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help=(
            "Optional minimum variance-of-Laplacian sharpness. "
            "Default 0 disables hard sharpness rejection."
        ),
    )
    parser.add_argument(
        "--bestshot-dir",
        default="outputs/day7_bestshots",
        help="Directory for final TOP-K plate crops",
    )
    parser.add_argument(
        "--save-plates",
        action="store_true",
        help="Also save every accepted raw plate crop to outputs/plates",
    )
    parser.add_argument(
        "--verbose-plates",
        action="store_true",
        help="Print plate size, sharpness, detector confidence and quality score",
    )

    # EasyOCR is kept in the Python 3.14 Day 5 environment. PaddleOCR can be run
    # later in the separate Python 3.12 environment on outputs/day7_bestshots.
    parser.add_argument(
        "--ocr-mode",
        choices=("bestshot", "off"),
        default="bestshot",
        help=(
            "bestshot: run EasyOCR only when a crop enters current TOP-K; "
            "off: collect best shots without OCR"
        ),
    )
    parser.add_argument(
        "--min-ocr-confidence",
        type=float,
        default=0.30,
        help="Minimum EasyOCR confidence used in plate aggregation",
    )
    parser.add_argument(
        "--text-similarity",
        type=float,
        default=0.65,
        help=(
            "Similarity threshold for fuzzy temporal aggregation of OCR strings "
            "from TOP-K candidates"
        ),
    )

    # Existing Day 4 counting line.
    parser.add_argument(
        "--count-line-y",
        type=int,
        default=500,
        help="Y coordinate of the counting / demo stop line",
    )

    # Optional homography / speed estimation.
    parser.add_argument(
        "--homography-config",
        default="configs/homography.json",
        help=(
            "JSON with src_points/world_points/homography. If it does not exist, "
            "speed estimation stays disabled unless --calibrate-homography is used."
        ),
    )
    parser.add_argument(
        "--calibrate-homography",
        action="store_true",
        help=(
            "Interactively click 4 road points in order: top-left, top-right, "
            "bottom-right, bottom-left, then save homography config"
        ),
    )
    parser.add_argument(
        "--road-width-m",
        type=float,
        default=10.0,
        help="Real/calibration width of selected road quadrilateral in meters",
    )
    parser.add_argument(
        "--road-length-m",
        type=float,
        default=30.0,
        help="Real/calibration length of selected road quadrilateral in meters",
    )
    parser.add_argument(
        "--speed-window-sec",
        type=float,
        default=0.50,
        help="Minimum time window used for speed estimation",
    )
    parser.add_argument(
        "--speed-median-window",
        type=int,
        default=5,
        help="Median smoothing window for speed estimates",
    )

    # Violation-logic prototypes. They intentionally reuse the existing image line.
    parser.add_argument(
        "--enable-red-light-demo",
        action="store_true",
        help="Enable simulated RED/GREEN state and red-light crossing events",
    )
    parser.add_argument(
        "--red-intervals",
        default="5-12,25-33",
        help="Simulated red intervals in seconds, e.g. 5-12,25-33",
    )
    parser.add_argument(
        "--red-direction",
        choices=("up", "down", "both"),
        default="down",
        help="Crossing direction that counts as red-light violation in the demo",
    )
    parser.add_argument(
        "--allowed-direction",
        choices=("up", "down", "both"),
        default="both",
        help=(
            "Prototype wrong-way rule. Example: --allowed-direction down marks "
            "UP crossings as wrong-way. 'both' disables wrong-way events."
        ),
    )
    parser.add_argument(
        "--events-output",
        default="outputs/day7_events.jsonl",
        help="JSONL output for violation prototype events",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show preview window; press q to stop",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ocr_every < 1:
        raise ValueError("--ocr-every must be >= 1")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_plate_width < 1 or args.min_plate_height < 1:
        raise ValueError("Minimum plate dimensions must be >= 1")
    if not 0.0 <= args.min_quality_plate_conf <= 1.0:
        raise ValueError("--min-quality-plate-conf must be in [0, 1]")
    if not 0.0 <= args.min_ocr_confidence <= 1.0:
        raise ValueError("--min-ocr-confidence must be in [0, 1]")
    if not 0.0 <= args.text_similarity <= 1.0:
        raise ValueError("--text-similarity must be in [0, 1]")
    if args.speed_window_sec <= 0:
        raise ValueError("--speed-window-sec must be > 0")
    if args.speed_median_window < 1:
        raise ValueError("--speed-median-window must be >= 1")
    if args.road_width_m <= 0 or args.road_length_m <= 0:
        raise ValueError("Road calibration dimensions must be > 0")


def calculate_sharpness(image: np.ndarray) -> float:
    """Variance of Laplacian: a simple heuristic sharpness measure."""
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_plate_quality(
    plate_crop: np.ndarray,
    plate_conf: float,
) -> tuple[float, float, int, int]:
    """
    Quality-score heuristic inherited from Day 6 and used in the Day 7 demo.

    It intentionally uses only visual/detector information so OCR is not required
    to decide whether the frame deserves to become a best-shot candidate.
    """
    h, w = plate_crop.shape[:2]
    sharpness = calculate_sharpness(plate_crop)
    area = w * h

    # Heuristic normalizations for the current CCTV experiment.
    size_score = min(area / 4000.0, 1.0)
    sharpness_score = min(sharpness / 300.0, 1.0)
    confidence_score = float(np.clip(plate_conf, 0.0, 1.0))

    score = (
        0.40 * sharpness_score
        + 0.35 * size_score
        + 0.25 * confidence_score
    )

    return float(score), float(sharpness), int(w), int(h)


def update_best_candidates(
    best_candidates: dict[int, list[PlateCandidate]],
    track_id: int,
    frame_idx: int,
    plate_crop: np.ndarray,
    plate_conf: float,
    top_k: int,
) -> tuple[PlateCandidate, bool]:
    """Insert candidate, keep TOP-K by visual quality, return whether it survived."""
    score, sharpness, width, height = calculate_plate_quality(
        plate_crop,
        plate_conf,
    )

    candidate = PlateCandidate(
        score=score,
        frame_idx=frame_idx,
        crop=plate_crop.copy(),
        plate_conf=float(plate_conf),
        sharpness=sharpness,
        width=width,
        height=height,
    )

    candidates = best_candidates[track_id]
    candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)

    if len(candidates) > top_k:
        del candidates[top_k:]

    accepted = any(item is candidate for item in candidates)
    return candidate, accepted


def normalize_plate_text(text: str) -> str:
    """Uppercase OCR text and keep only Latin letters and digits."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_plausible_plate(text: str) -> bool:
    """
    Soft baseline filter only.

    Exact Kyrgyz plate masks should be validated later on a confirmed specification
    and ground-truth dataset. Here we only reject obvious OCR garbage.
    """
    if not text or not text.isalnum():
        return False
    if len(text) < 5 or len(text) > 12:
        return False

    has_digit = any(char.isdigit() for char in text)
    has_letter = any(char.isalpha() for char in text)
    return has_digit and has_letter


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """Gentle OCR preprocessing: upscale small crop, grayscale and CLAHE."""
    if image is None or image.size == 0:
        return image

    _, width = image.shape[:2]

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
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    return clahe.apply(gray)


def recognize_plate(
    ocr_reader: easyocr.Reader,
    image: np.ndarray,
) -> tuple[str, float]:
    """Run EasyOCR and merge text fragments left-to-right."""
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

    ocr_results = sorted(
        ocr_results,
        key=lambda item: min(point[0] for point in item[0]),
    )

    fragments: list[str] = []
    weighted_confidence = 0.0
    total_characters = 0

    for _, text, confidence in ocr_results:
        fragment = normalize_plate_text(text)
        if not fragment:
            continue

        fragments.append(fragment)
        fragment_length = len(fragment)
        weighted_confidence += float(confidence) * fragment_length
        total_characters += fragment_length

    if not fragments or total_characters == 0:
        return "", 0.0

    merged = "".join(fragments)
    confidence = weighted_confidence / total_characters
    return merged, float(confidence)


def levenshtein_distance(a: str, b: str) -> int:
    """Small dependency-free edit distance for fuzzy temporal aggregation."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def text_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / longest


def aggregate_plate_from_candidates(
    candidates: list[PlateCandidate],
    min_ocr_confidence: float,
    similarity_threshold: float,
) -> str:
    """
    Fuzzy temporal aggregation over OCR results of current TOP-K candidates.

    Similar strings are clustered using normalized Levenshtein similarity. The
    winning cluster is the one with the largest accumulated quality-weighted OCR
    confidence. Its representative is the strongest individual observation.
    """
    observations = [
        candidate
        for candidate in candidates
        if (
            is_plausible_plate(candidate.plate_text)
            and candidate.ocr_conf >= min_ocr_confidence
        )
    ]

    if not observations:
        return ""

    clusters: list[dict[str, object]] = []

    for candidate in observations:
        weight = float(candidate.ocr_conf) * (0.5 + 0.5 * candidate.score)
        matched_cluster = None

        for cluster in clusters:
            representative = str(cluster["representative"])
            if text_similarity(candidate.plate_text, representative) >= similarity_threshold:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append(
                {
                    "representative": candidate.plate_text,
                    "score": weight,
                    "best_weight": weight,
                }
            )
            continue

        matched_cluster["score"] = float(matched_cluster["score"]) + weight
        if weight > float(matched_cluster["best_weight"]):
            matched_cluster["representative"] = candidate.plate_text
            matched_cluster["best_weight"] = weight

    best_cluster = max(
        clusters,
        key=lambda cluster: float(cluster["score"]),
    )
    return str(best_cluster["representative"])


def save_best_candidates(
    best_candidates: dict[int, list[PlateCandidate]],
    output_dir: str | Path,
) -> int:
    """Save final TOP-K crops for every track and return number of files written."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for track_id, candidates in sorted(best_candidates.items()):
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)

        for rank, candidate in enumerate(ranked, start=1):
            text_suffix = (
                f"_{candidate.plate_text}"
                if candidate.plate_text
                else ""
            )
            filename = output_dir / (
                f"track_{track_id:04d}"
                f"_rank_{rank}"
                f"_frame_{candidate.frame_idx:06d}"
                f"_q_{candidate.score:.3f}"
                f"_sharp_{candidate.sharpness:.1f}"
                f"_det_{candidate.plate_conf:.2f}"
                f"{text_suffix}.jpg"
            )
            cv2.imwrite(str(filename), candidate.crop)
            saved += 1

    return saved


def parse_red_intervals(value: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    value = value.strip()

    if not value:
        return intervals

    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        parts = chunk.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid red interval '{chunk}'. Expected format start-end"
            )

        start = float(parts[0])
        end = float(parts[1])

        if end < start:
            start, end = end, start

        intervals.append((start, end))

    return intervals


def get_light_state(
    timestamp: float,
    red_intervals: list[tuple[float, float]],
) -> str:
    for start, end in red_intervals:
        if start <= timestamp <= end:
            return "RED"
    return "GREEN"


def direction_matches(rule: str, direction: str) -> bool:
    return rule == "both" or rule == direction


def load_homography_config(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    homography = np.array(data["homography"], dtype=np.float64)
    src_points = np.array(data["src_points"], dtype=np.float32)

    if homography.shape != (3, 3):
        raise ValueError("Homography matrix must have shape 3x3")
    if src_points.shape != (4, 2):
        raise ValueError("src_points must contain exactly four 2D points")

    return homography, src_points, data


def calibrate_homography(
    frame: np.ndarray,
    road_width_m: float,
    road_length_m: float,
    output_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interactive four-point road calibration.

    Click in this exact order:
    1) top-left, 2) top-right, 3) bottom-right, 4) bottom-left.
    """
    points: list[list[float]] = []
    window_name = "Day 7 Homography Calibration"

    def mouse_callback(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([float(x), float(y)])
            print(f"[CALIBRATION] Point {len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("[CALIBRATION] Click 4 points:")
    print("[CALIBRATION] 1 top-left -> 2 top-right -> 3 bottom-right -> 4 bottom-left")
    print("[CALIBRATION] Press R to reset, ESC/Q to cancel")

    while len(points) < 4:
        display = frame.copy()

        cv2.putText(
            display,
            "Click: TL -> TR -> BR -> BL | R reset | Q/ESC cancel",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        for index, (x, y) in enumerate(points, start=1):
            x_i, y_i = int(x), int(y)
            cv2.circle(display, (x_i, y_i), 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(index),
                (x_i + 8, y_i - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        if len(points) >= 2:
            poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [poly], False, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Homography calibration cancelled")
        if key in (ord("r"), ord("R")):
            points.clear()
            print("[CALIBRATION] Points reset")

    cv2.destroyWindow(window_name)

    src_points = np.array(points, dtype=np.float32)
    world_points = np.array(
        [
            [0.0, 0.0],
            [road_width_m, 0.0],
            [road_width_m, road_length_m],
            [0.0, road_length_m],
        ],
        dtype=np.float32,
    )

    homography, status = cv2.findHomography(src_points, world_points)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "src_points": src_points.tolist(),
        "world_points": world_points.tolist(),
        "homography": homography.tolist(),
        "road_width_m": float(road_width_m),
        "road_length_m": float(road_length_m),
        "note": (
            "Day 7 demo calibration. Speed is an estimate unless real road dimensions "
            "and a validated calibration procedure are used."
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"[CALIBRATION] Saved: {output_path}")
    return homography.astype(np.float64), src_points


def point_inside_polygon(
    point: tuple[int, int],
    polygon: np.ndarray,
) -> bool:
    contour = polygon.astype(np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(contour, point, False) >= 0


def pixel_to_world(
    point: tuple[int, int],
    homography: np.ndarray,
) -> tuple[float, float]:
    array = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(array, homography)
    world_x, world_y = transformed[0][0]
    return float(world_x), float(world_y)


def estimate_speed_kmh(
    history: deque[tuple[float, float, float]],
    min_dt: float,
) -> float | None:
    """Estimate speed over a temporal window instead of adjacent jittery frames."""
    if len(history) < 2:
        return None

    current_t, current_x, current_y = history[-1]
    previous = None

    # Choose the most recent point that is at least min_dt seconds old.
    for item in reversed(list(history)[:-1]):
        old_t, _, _ = item
        if current_t - old_t >= min_dt:
            previous = item
            break

    if previous is None:
        return None

    old_t, old_x, old_y = previous
    dt = current_t - old_t

    if dt <= 0:
        return None

    distance_m = float(np.hypot(current_x - old_x, current_y - old_y))
    return float((distance_m / dt) * 3.6)


def append_event(path: str | Path, event: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    red_intervals = parse_red_intervals(args.red_intervals)

    source_path = Path(args.source)
    plate_model_path = Path(args.plate_model)

    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")
    if not plate_model_path.exists():
        raise FileNotFoundError(f"Plate model not found: {plate_model_path}")

    print(f"[INFO] Loading vehicle model: {args.model}")
    vehicle_model = YOLO(args.model)

    print(f"[INFO] Loading plate model: {args.plate_model}")
    plate_model = YOLO(args.plate_model)

    ocr_reader = None
    if args.ocr_mode == "bestshot":
        print("[INFO] Loading EasyOCR (CPU)")
        ocr_reader = easyocr.Reader(["en"], gpu=False)
        print("[INFO] EasyOCR loaded")
    else:
        print("[INFO] OCR disabled. Best shots will still be collected.")

    print(f"[INFO] Opening video: {source_path}")
    cap = cv2.VideoCapture(str(source_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 1:
        fps = 25.0
        print("[WARNING] Invalid source FPS; using fallback 25 FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps:.2f}")

    count_line_y = args.count_line_y
    if count_line_y >= height or count_line_y < 0:
        old_value = count_line_y
        count_line_y = int(height * 0.70)
        print(
            f"[WARNING] count-line-y={old_value} is outside video height={height}; "
            f"using {count_line_y}"
        )

    # Optional homography setup.
    homography: np.ndarray | None = None
    calibration_polygon: np.ndarray | None = None
    homography_path = Path(args.homography_config)

    if args.calibrate_homography:
        success, calibration_frame = cap.read()
        if not success:
            cap.release()
            raise RuntimeError("Could not read first frame for homography calibration")

        homography, calibration_polygon = calibrate_homography(
            calibration_frame,
            args.road_width_m,
            args.road_length_m,
            homography_path,
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif homography_path.exists():
        try:
            homography, calibration_polygon, _ = load_homography_config(
                homography_path
            )
            print(f"[INFO] Loaded homography: {homography_path}")
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            print(f"[WARNING] Could not load homography config: {error}")
            print("[WARNING] Speed estimation disabled")
            homography = None
            calibration_polygon = None
    else:
        print(
            f"[INFO] Homography config not found: {homography_path}. "
            "Speed estimation disabled. Use --calibrate-homography to create it."
        )

    # Output video.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    raw_plate_dir = Path("outputs/plates")
    if args.save_plates:
        raw_plate_dir.mkdir(parents=True, exist_ok=True)

    events_path = Path(args.events_output)
    if events_path.exists():
        # Avoid mixing events from an older run with the current run.
        events_path.unlink()

    # Track / ANPR / geometry state.
    track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)
    seen_ids: set[int] = set()
    counted_up_ids: set[int] = set()
    counted_down_ids: set[int] = set()

    best_candidates: dict[int, list[PlateCandidate]] = defaultdict(list)
    best_plate_by_id: dict[int, str] = {}

    world_history: dict[int, deque[tuple[float, float, float]]] = defaultdict(
        lambda: deque(maxlen=max(int(fps * 3), 30))
    )
    speed_history: dict[int, deque[float]] = defaultdict(
        lambda: deque(maxlen=args.speed_median_window)
    )
    current_speed_by_id: dict[int, float] = {}

    red_light_violation_ids: set[int] = set()
    wrong_way_ids: set[int] = set()
    event_banner_until: dict[int, tuple[int, str]] = {}

    frame_number = 0
    up_count = 0
    down_count = 0
    plate_candidates_seen = 0
    plate_candidates_accepted = 0
    ocr_calls = 0

    # Benchmark only the video-processing pipeline.
    # Model/OCR initialization above is intentionally excluded.
    pipeline_start = time.perf_counter()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            timestamp = frame_number / fps

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

            # Existing Day 4 count line, reused as a simple demo stop line.
            cv2.line(
                annotated_frame,
                (0, count_line_y),
                (width, count_line_y),
                (0, 255, 255),
                3,
            )
            cv2.putText(
                annotated_frame,
                "COUNT / DEMO STOP LINE",
                (20, max(count_line_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            if calibration_polygon is not None:
                cv2.polylines(
                    annotated_frame,
                    [calibration_polygon.astype(np.int32).reshape((-1, 1, 2))],
                    True,
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    "SPEED CALIBRATION ROI",
                    (20, min(height - 20, 230)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

            light_state = "N/A"
            if args.enable_red_light_demo:
                light_state = get_light_state(timestamp, red_intervals)
                light_color = (
                    (0, 0, 255)
                    if light_state == "RED"
                    else (0, 255, 0)
                )
                cv2.putText(
                    annotated_frame,
                    f"LIGHT DEMO: {light_state}",
                    (20, min(height - 20, 265)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    light_color,
                    2,
                )

            run_plate_detection = frame_number % args.ocr_every == 0

            if (
                result.boxes is not None
                and result.boxes.is_track
                and result.boxes.id is not None
            ):
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
                    # Best-shot ANPR (integrated Day 7 demo)
                    # -------------------------------------------------
                    vehicle_crop = frame[y1:y2, x1:x2]

                    if run_plate_detection and vehicle_crop.size > 0:
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
                                plate_confidences = (
                                    plate_result.boxes.conf.detach().cpu().numpy()
                                )
                                best_index = int(np.argmax(plate_confidences))
                                plate_box = (
                                    plate_result.boxes.xyxy[best_index]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                                plate_detection_confidence = float(
                                    plate_confidences[best_index]
                                )

                                px1, py1, px2, py2 = map(int, plate_box)
                                vehicle_h, vehicle_w = vehicle_crop.shape[:2]

                                px1 = max(0, px1)
                                py1 = max(0, py1)
                                px2 = min(vehicle_w, px2)
                                py2 = min(vehicle_h, py2)

                                if px2 > px1 and py2 > py1:
                                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                                    if plate_crop.size > 0:
                                        plate_h, plate_w = plate_crop.shape[:2]
                                        sharpness = calculate_sharpness(plate_crop)
                                        plate_candidates_seen += 1

                                        # Full-frame plate rectangle for diagnostics.
                                        global_px1 = x1 + px1
                                        global_py1 = y1 + py1
                                        global_px2 = x1 + px2
                                        global_py2 = y1 + py2

                                        cv2.rectangle(
                                            annotated_frame,
                                            (global_px1, global_py1),
                                            (global_px2, global_py2),
                                            (0, 255, 255),
                                            2,
                                        )

                                        passes_filter = (
                                            plate_w >= args.min_plate_width
                                            and plate_h >= args.min_plate_height
                                            and plate_detection_confidence
                                            >= args.min_quality_plate_conf
                                            and sharpness >= args.min_sharpness
                                        )

                                        if args.save_plates and passes_filter:
                                            raw_plate_file = raw_plate_dir / (
                                                f"frame_{frame_number:06d}_"
                                                f"id_{track_id}.jpg"
                                            )
                                            cv2.imwrite(
                                                str(raw_plate_file),
                                                plate_crop,
                                            )

                                        if passes_filter:
                                            candidate, accepted = update_best_candidates(
                                                best_candidates,
                                                track_id,
                                                frame_number,
                                                plate_crop,
                                                plate_detection_confidence,
                                                args.top_k,
                                            )

                                            if accepted:
                                                plate_candidates_accepted += 1

                                                if args.ocr_mode == "bestshot":
                                                    assert ocr_reader is not None
                                                    processed = preprocess_plate(
                                                        candidate.crop
                                                    )
                                                    text, ocr_conf = recognize_plate(
                                                        ocr_reader,
                                                        processed,
                                                    )
                                                    ocr_calls += 1
                                                    candidate.plate_text = (
                                                        normalize_plate_text(text)
                                                    )
                                                    candidate.ocr_conf = float(ocr_conf)

                                                best_plate_by_id[track_id] = (
                                                    aggregate_plate_from_candidates(
                                                        best_candidates[track_id],
                                                        args.min_ocr_confidence,
                                                        args.text_similarity,
                                                    )
                                                )

                                            if args.verbose_plates:
                                                score, _, _, _ = calculate_plate_quality(
                                                    plate_crop,
                                                    plate_detection_confidence,
                                                )
                                                print(
                                                    f"[PLATE] frame={frame_number} "
                                                    f"ID={track_id} "
                                                    f"size={plate_w}x{plate_h} "
                                                    f"sharp={sharpness:.1f} "
                                                    f"det={plate_detection_confidence:.2f} "
                                                    f"quality={score:.3f} "
                                                    f"accepted={accepted if passes_filter else False}"
                                                )

                    # -------------------------------------------------
                    # Existing Day 4 bottom-center trajectory
                    # -------------------------------------------------
                    center_x = int((x1 + x2) / 2)
                    center_y = y2

                    history = track_history[track_id]
                    previous_y = history[-1][1] if history else None
                    history.append((center_x, center_y))
                    if len(history) > 30:
                        history.pop(0)

                    crossed_down = False
                    crossed_up = False

                    if previous_y is not None:
                        crossed_down = previous_y < count_line_y <= center_y
                        crossed_up = previous_y > count_line_y >= center_y

                        if crossed_down and track_id not in counted_down_ids:
                            counted_down_ids.add(track_id)
                            down_count += 1
                            print(
                                f"[COUNT DOWN] {class_name} "
                                f"ID:{track_id} Down:{down_count}"
                            )

                        elif crossed_up and track_id not in counted_up_ids:
                            counted_up_ids.add(track_id)
                            up_count += 1
                            print(
                                f"[COUNT UP] {class_name} "
                                f"ID:{track_id} Up:{up_count}"
                            )

                    # -------------------------------------------------
                    # Experimental homography -> rough speed estimate
                    # -------------------------------------------------
                    if homography is not None and calibration_polygon is not None:
                        point = (center_x, center_y)

                        if point_inside_polygon(point, calibration_polygon):
                            world_x, world_y = pixel_to_world(point, homography)
                            track_world_history = world_history[track_id]
                            track_world_history.append(
                                (timestamp, world_x, world_y)
                            )

                            raw_speed = estimate_speed_kmh(
                                track_world_history,
                                args.speed_window_sec,
                            )

                            if raw_speed is not None and np.isfinite(raw_speed):
                                # Reject clearly nonsensical extrapolation spikes in the
                                # demo overlay; calibration quality still determines truth.
                                if 0.0 <= raw_speed <= 250.0:
                                    speed_history[track_id].append(raw_speed)
                                    current_speed_by_id[track_id] = float(
                                        np.median(speed_history[track_id])
                                    )
                        else:
                            # Do not bridge a speed estimate across time spent outside ROI.
                            world_history[track_id].clear()
                            speed_history[track_id].clear()
                            current_speed_by_id.pop(track_id, None)

                    # -------------------------------------------------
                    # Experimental violation prototypes using existing crossing event
                    # -------------------------------------------------
                    crossing_direction = None
                    if crossed_down:
                        crossing_direction = "down"
                    elif crossed_up:
                        crossing_direction = "up"

                    if crossing_direction is not None:
                        best_plate = best_plate_by_id.get(track_id, "")
                        speed_value = current_speed_by_id.get(track_id)

                        if (
                            args.enable_red_light_demo
                            and light_state == "RED"
                            and direction_matches(
                                args.red_direction,
                                crossing_direction,
                            )
                            and track_id not in red_light_violation_ids
                        ):
                            red_light_violation_ids.add(track_id)
                            event = {
                                "event_type": "red_light_demo",
                                "track_id": int(track_id),
                                "direction": crossing_direction,
                                "timestamp_sec": round(timestamp, 3),
                                "plate": best_plate or None,
                                "speed_kmh": (
                                    round(float(speed_value), 1)
                                    if speed_value is not None
                                    else None
                                ),
                            }
                            append_event(events_path, event)
                            event_banner_until[track_id] = (
                                frame_number + int(fps * 1.5),
                                "RED LIGHT DEMO",
                            )
                            print(f"[EVENT] {event}")

                        wrong_way = (
                            args.allowed_direction != "both"
                            and crossing_direction != args.allowed_direction
                        )

                        if wrong_way and track_id not in wrong_way_ids:
                            wrong_way_ids.add(track_id)
                            event = {
                                "event_type": "wrong_way_demo",
                                "track_id": int(track_id),
                                "direction": crossing_direction,
                                "allowed_direction": args.allowed_direction,
                                "timestamp_sec": round(timestamp, 3),
                                "plate": best_plate or None,
                                "speed_kmh": (
                                    round(float(speed_value), 1)
                                    if speed_value is not None
                                    else None
                                ),
                            }
                            append_event(events_path, event)
                            event_banner_until[track_id] = (
                                frame_number + int(fps * 1.5),
                                "WRONG WAY DEMO",
                            )
                            print(f"[EVENT] {event}")

                    # -------------------------------------------------
                    # Vehicle visualization
                    # -------------------------------------------------
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    best_plate = best_plate_by_id.get(track_id, "")
                    speed_value = current_speed_by_id.get(track_id)

                    label_parts = [
                        class_name,
                        f"ID:{track_id}",
                    ]
                    if best_plate:
                        label_parts.append(best_plate)
                    if speed_value is not None:
                        label_parts.append(f"{speed_value:.1f} km/h")
                    label_parts.append(f"{confidence:.2f}")

                    cv2.putText(
                        annotated_frame,
                        " ".join(label_parts),
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    cv2.circle(
                        annotated_frame,
                        (center_x, center_y),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                    points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                    if len(points) > 1:
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(255, 255, 0),
                            thickness=2,
                        )

                    banner = event_banner_until.get(track_id)
                    if banner is not None:
                        until_frame, text = banner
                        if frame_number <= until_frame:
                            cv2.putText(
                                annotated_frame,
                                text,
                                (x1, min(y2 + 25, height - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                        else:
                            event_banner_until.pop(track_id, None)

            total_count = up_count + down_count

            # Global statistics.
            stats = [
                f"Frame: {frame_number}",
                f"Active tracks: {active_tracks}",
                f"Unique vehicle IDs: {len(seen_ids)}",
                f"UP: {up_count}",
                f"DOWN: {down_count}",
                f"TOTAL: {total_count}",
                f"Best-shot tracks: {len(best_candidates)}",
                f"OCR calls: {ocr_calls}",
            ]

            for index, text in enumerate(stats):
                cv2.putText(
                    annotated_frame,
                    text,
                    (20, 30 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 if index < 3 else 0.72,
                    (255, 255, 255) if index != 5 else (0, 255, 255),
                    2,
                )

            writer.write(annotated_frame)

            if args.show:
                cv2.imshow("TrafficVision KG - Day 7 Demo", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    saved_bestshots = save_best_candidates(
        best_candidates,
        args.bestshot_dir,
    )

    # Final re-aggregation guarantees labels match the final TOP-K, not an earlier set.
    if args.ocr_mode == "bestshot":
        for track_id, candidates in best_candidates.items():
            best_plate_by_id[track_id] = aggregate_plate_from_candidates(
                candidates,
                args.min_ocr_confidence,
                args.text_similarity,
            )

    recognized_tracks = sum(bool(text) for text in best_plate_by_id.values())
    total_count = up_count + down_count

    pipeline_elapsed = time.perf_counter() - pipeline_start
    pipeline_fps = (
        frame_number / pipeline_elapsed
        if pipeline_elapsed > 0.0
        else 0.0
    )

    print()
    print("[DONE] TrafficVision KG Day 7 demo finished")
    print(f"[DONE] Output video: {output_path}")
    print(f"[DONE] Best-shot directory: {args.bestshot_dir}")
    print(f"[DONE] Best-shot files saved: {saved_bestshots}")
    print(f"[DONE] Plate candidates seen: {plate_candidates_seen}")
    print(f"[DONE] Candidates that entered TOP-K: {plate_candidates_accepted}")
    print(f"[DONE] EasyOCR calls: {ocr_calls}")
    print(f"[DONE] Unique vehicle IDs: {len(seen_ids)}")
    print(f"[DONE] Vehicles UP: {up_count}")
    print(f"[DONE] Vehicles DOWN: {down_count}")
    print(f"[DONE] Total counted vehicles: {total_count}")
    print(f"[DONE] Frames processed: {frame_number}")
    print(f"[DONE] Processing time: {pipeline_elapsed:.2f} sec")
    print(f"[DONE] Pipeline FPS: {pipeline_fps:.2f}")
    print(f"[DONE] Tracks with plausible aggregated plate: {recognized_tracks}")
    print(f"[DONE] Red-light demo events: {len(red_light_violation_ids)}")
    print(f"[DONE] Wrong-way demo events: {len(wrong_way_ids)}")

    if homography is None:
        print("[DONE] Speed: disabled (no homography config)")
    else:
        print("[DONE] Speed: enabled as rough/calibration-dependent estimate")

    if args.ocr_mode == "bestshot":
        print(
            "[NEXT] For PaddleOCR recognition-only, run your Python 3.12 Day 5 "
            f"test against: {args.bestshot_dir}"
        )


if __name__ == "__main__":
    main()

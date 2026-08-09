from pathlib import Path

from paddleocr import TextRecognition


PLATES_DIR = Path("outputs/plates")

# Сначала пробуем ту же recognition-модель,
# которую PaddleOCR 3.7 уже использовал в полном pipeline.
MODEL_NAME = "PP-OCRv6_medium_rec"


def extract_result(result):
    payload = result.json
    data = payload.get("res", payload)

    text = str(data.get("rec_text", "")).strip()
    score = float(data.get("rec_score", 0.0))

    return text, score


def main():
    print("[INFO] Loading Paddle text recognizer...")
    print(f"[INFO] Model: {MODEL_NAME}")

    try:
        model = TextRecognition(
            model_name=MODEL_NAME,
            device="cpu",
        )
    except Exception as exc:
        print(
            f"[WARNING] Could not load {MODEL_NAME}: "
            f"{type(exc).__name__}: {exc}"
        )
        print("[INFO] Falling back to PP-OCRv5_server_rec")

        model = TextRecognition(
            model_name="PP-OCRv5_server_rec",
            device="cpu",
        )

    image_paths = sorted(
        list(PLATES_DIR.glob("*.jpg"))
        + list(PLATES_DIR.glob("*.jpeg"))
        + list(PLATES_DIR.glob("*.png"))
    )

    if not image_paths:
        raise RuntimeError(
            f"No plate images found in: {PLATES_DIR.resolve()}"
        )

    print(f"[INFO] Found {len(image_paths)} plate crops")

    recognized = 0
    strong = 0

    for image_path in image_paths:
        try:
            results = model.predict(
                input=str(image_path),
                batch_size=1,
            )
        except Exception as exc:
            print(
                f"[ERROR] {image_path.name} "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if not results:
            print(
                f"[REC] {image_path.name} NO RESULT"
            )
            continue

        text, score = extract_result(results[0])

        if text:
            recognized += 1

        if text and score >= 0.50:
            strong += 1

        print(
            f"[REC] "
            f"{image_path.name} "
            f"text={text!r} "
            f"conf={score:.2f}"
        )

    print()
    print("[DONE] Recognition-only test finished")
    print(f"[DONE] Total crops: {len(image_paths)}")
    print(f"[DONE] Non-empty predictions: {recognized}")
    print(f"[DONE] Predictions conf>=0.50: {strong}")


if __name__ == "__main__":
    main()

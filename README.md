# TrafficVision-KG

Прототип системы видеоаналитики транспорта для подготовки
к собеседованию на позицию Computer Vision Engineer.

## День 1

Реализована детекция транспортных средств на видео:

- car;
- motorcycle;
- bus;
- truck.

## Стек

- Python
- OpenCV
- Ultralytics YOLO11
- PyTorch

## Запуск

```bash
python src/detect_vehicles.py \
  --source data/input/intersection.mp4 \
  --output data/output/result.mp4 \
  --conf 0.25 \
  --iou 0.70 \
  --imgsz 640

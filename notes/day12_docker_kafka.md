# Day 12 — Docker + Kafka

## Цель

Добавить messaging layer к TrafficVision-KG после DeepStream Day 11:

    DeepStream / CV
          ↓
    TrafficVision JSON event
          ↓
    Kafka
          ↓
    Backend consumer

Главная задача — отделить edge CV pipeline от backend processing.

## Что сделано

- Поднят Apache Kafka 4.3.1 через Docker Compose.
- Создан topic `trafficvision.events`.
- Создано 4 partitions.
- Написан `src/messaging/producer.py`.
- Написан `src/messaging/consumer.py`.
- Использован `confluent-kafka 2.15.0`.
- Consumer запускается в отдельном lightweight Docker container.
- Добавлен JSON event contract с:
  - `event_id`
  - `camera_id`
  - `source_id`
  - `track_id`
  - vehicle class
  - ANPR / speed / violation metadata
- Consumer использует manual offset commit.
- Events сохраняются в `outputs/day12/consumed_events.jsonl`.

## Среда

- Jetson Orin Nano
- Docker 29.7.1
- Docker Compose 5.3.1
- Apache Kafka 4.3.1
- Python 3.12.14
- confluent-kafka 2.15.0

## Event example

    event_type: vehicle_observation
    camera_id: demo-cam-00
    source_id: 0
    track_id: 14
    vehicle_class: car

Kafka key:

    camera_id:track_id

Пример:

    demo-cam-00:14

ANPR, speed и violation пока не считаются валидированными, поэтому соответствующие поля передаются как `null` / `validated=false`.

## Failure / recovery test

Проведён тест с остановкой backend consumer.

До остановки:

    TOTAL LAG = 0

Consumer был остановлен, после чего отправлено 5 новых events.

Kafka продолжила принимать сообщения.

Результат:

    partition 0: LAG = 0
    partition 1: LAG = 2
    partition 2: LAG = 1
    partition 3: LAG = 2

    TOTAL LAG = 5

После повторного запуска consumer:

    partition 0: LAG = 0
    partition 1: LAG = 0
    partition 2: LAG = 0
    partition 3: LAG = 0

    TOTAL LAG = 0

Это подтвердило, что backend может временно быть недоступен, а Kafka сохраняет backlog до восстановления consumer.

## Resource usage

`docker stats --no-stream`:

Kafka:

    CPU: ~1.30 %
    RAM: ~951.5 MiB

Consumer:

    CPU: ~0.10 %
    RAM: ~16.36 MiB

Kafka broker заметно тяжелее Python consumer, поэтому в production логичнее рассматривать Jetson как edge producer, а Kafka — как центральную инфраструктуру.

## Проблемы

### Docker build timeout

Первый build упал при ненужном обновлении pip:

    ReadTimeoutError
    files.pythonhosted.org

Решение:

- убрать `pip install --upgrade pip`;
- увеличить pip timeout;
- добавить retries.

После этого consumer image успешно собрался.

### Docker ownership

Consumer запускается с host UID/GID.

Файл:

    outputs/day12/consumed_events.jsonl

создан с ownership:

    1000:1000

Проблема Day 11 с root-owned bind-mounted файлами не повторилась.

## Итог

Day 12 добавил первый production-style messaging layer:

    Edge CV
      ↓
    JSON event
      ↓
    Kafka
      ↓
    Backend consumer

Главный результат:

    backend DOWN → TOTAL LAG = 5
    backend UP   → TOTAL LAG = 0

Следующий логичный шаг — связать DeepStream metadata с Kafka через event logic / `nvmsgconv` / `nvmsgbroker`.

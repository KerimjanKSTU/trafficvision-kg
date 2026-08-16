# TrafficVision-KG — Day 12: Docker + Kafka

## 1. Цель

Добавить следующий production-слой после DeepStream Day 11:

```text
DeepStream / CV pipeline
        ↓
TrafficVision JSON event
        ↓
Kafka Producer
        ↓
Kafka Broker
        ↓
Consumer Group
        ↓
Backend / storage


---

## 25. Зафиксированная среда Day 12

Фактические версии на Jetson Orin Nano:

```text
Docker:           29.7.1
Docker Compose:   5.3.1
Apache Kafka:     4.3.1
Python:           3.12.14
confluent-kafka:  2.15.0

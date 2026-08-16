FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=300

COPY docker/day12/requirements.txt /tmp/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --retries 10 \
    -r /tmp/requirements.txt

COPY src/messaging /app/src/messaging

CMD ["python", "src/messaging/consumer.py"]

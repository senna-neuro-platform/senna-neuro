# ADR-0010: Metrics Contract and Export Path

- Status: Accepted
- Date: 2026-03-07

## Context

Для шага 12 нужен стабильный и тестируемый контур наблюдаемости: сбор метрик в ядре, экспорт в Prometheus и визуализация в Grafana.

## Decision

1. Сбор метрик выполняется в C++ через `MetricsCollector` (`src/core/metrics/metrics_collector.h`).
2. `MetricsCollector` не делает IO; он только агрегирует снапшот и отдаёт карту `senna_*` метрик.
3. Экспорт в Prometheus выполняется в Python (`infra/simulator/simulator_server.py`) через HTTP `/metrics`.
4. Exporter умеет читать runtime-снапшот из JSON-файла (`METRICS_SNAPSHOT_PATH`) и имеет synthetic fallback.
5. Метрика длительности такта публикуется как histogram (`senna_tick_duration_seconds`).
6. Дашборды Grafana провижатся из репозитория как JSON и поддерживаются в количестве трёх: Activity, Training, Performance.

## Consequences

- Ядро остаётся изолированным от сетевого/HTTP слоя.
- Контракт метрик унифицирован (`senna_*`) для C++, Prometheus и Grafana.
- В CI/локально возможна проверка формата exporter без запуска полного стенда.

# ADR-0006: Observability and Runtime Topology

Status: Accepted  
Date: 2026-03-07

## Context

Нужно зафиксировать стандартный runtime-контур для локального запуска и диагностики.

## Decision

1. Используем Docker Compose как стандартный способ подъема окружения.
2. Базовые сервисы: `simulator`, `prometheus`, `grafana`, `visualizer`.
3. Метрики экспортируются в формате Prometheus, графики строятся в Grafana.
4. `simulator` и `visualizer` читают только реальные артефакты из общего `data/artifacts`; при отсутствии snapshot/trace сервисы честно переходят в режим ожидания без synthetic fallback.

## Consequences

- Воспроизводимый локальный запуск одной командой.
- Единый формат наблюдаемости для разработки и CI-диагностики.
- Ручная и автоматическая приемка не маскируется искусственными данными.

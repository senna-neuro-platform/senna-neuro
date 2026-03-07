# ADR-0011: Python Bindings and Training Contract

- Status: Accepted
- Date: 2026-03-07

## Context

Шаг 14 добавляет Python-слой над C++ ядром: нужен стабильный API для обучения, метрик, сохранения/загрузки состояния и робастности.

## Decision

1. Контракт Python API фиксируется в pybind11-модуле `senna_core` (`src/bindings/python_module.cpp`).
2. Обязательные операции контракта: `create_network`, `load_sample`, `step`, `get_prediction`, `get_metrics`, `save_state`, `load_state`, `inject_noise`, `remove_neurons`.
3. Супервизия обучения реализуется отдельной операцией `supervise(expected_label)` через teacher-spike на правильный выходной нейрон.
4. YAML конфигурация валидируется только на стороне C++ (через `yaml-cpp`), Python-слой не дублирует валидацию.
5. Python training pipeline (`python/senna/training.py`, `python/train.py`) использует контракт модуля и поддерживает MNIST через `torchvision` с синтетическим fallback.
6. Интеграционные Python-тесты запускаются в CTest через `pytest` при доступности `pytest`.

## Consequences

- Интерфейс между C++ и Python становится явным и проверяемым.
- Основные ошибки конфигурации обнаруживаются на границе C++ до запуска обучения.
- Скрипт обучения и тесты можно запускать локально и в CI единообразно через CMake/CTest.

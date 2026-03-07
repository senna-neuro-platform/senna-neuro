# Step 16 Acceptance Runbook

Цель: пройти проверку DoD MVP (шаг 16) от развёртывания до итогового PASS/FAIL.

## Что входит

- Пошаговый сценарий запуска.
- Автоматизация ключевых проверок.
- Явные критерии PASS/FAIL.

## Файлы

- `docs/step16-acceptance/scripts/run_acceptance.sh`
- `docs/step16-acceptance/scripts/check_dod_metrics.py`
- `docs/step16-acceptance/scripts/check_inference_pipeline.py`
- `docs/step16-acceptance/scripts/check_ws_sparsity.py`

## Быстрый старт

```bash
cd /home/dsb/repos/github/senna-neuro-platform/senna-neuro
chmod +x docs/step16-acceptance/scripts/run_acceptance.sh
docs/step16-acceptance/scripts/run_acceptance.sh
```

Скрипт выполнит:

1. `make install`, `make build-debug`, `make test`
2. `make lint`
3. `make build-sanitize` + `ctest --preset sanitize`
4. `make up` + health-check endpoints
5. полный train-run (по умолчанию MNIST 60k/10k)
6. проверку метрик DoD (`accuracy`, `robustness`, `active_ratio`)
7. проверку pipeline inference (`MNIST image -> class 0..9`)
8. проверку sparsity по WebSocket (`activeCount/totalNeurons < 0.05`)

## Памятки Наблюдения Между Шагами

1. После `make up`: проверь состояние контейнеров через `docker compose ps`, логи через `make logs`, и что открываются `http://localhost:3000`, `http://localhost:8080`, `http://localhost:8000/metrics`.
2. Во время training-run: смотри `tail -f data/artifacts/training/metrics.jsonl`; параллельно в Grafana открой `SENNA Training` и `SENNA Activity`.
3. После проверки DoD-метрик: сверяй `eval_accuracy`, `active_ratio`, `prune_drop`, `noise_drop` в JSONL с графиками Grafana, чтобы подтвердить совпадение телеметрии.
4. После WS-проверки sparsity: открой `http://localhost:8080`, включи heatmap и покадровый режим `Next Tick`, визуально проверь волну и разреженность.

## Как Запускать Training-Run

Вариант A: через orchestrator (рекомендуется)

```bash
docs/step16-acceptance/scripts/run_acceptance.sh \
  --skip-build \
  --skip-lint \
  --skip-sanitize \
  --skip-docker \
  --skip-ws-sparsity \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

Вариант B: напрямую `python/train.py`

```bash
make install
make build-debug
PYTHONPATH=build/debug:python python3 python/train.py \
  --config configs/default.yaml \
  --dataset mnist \
  --data-root data \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --ticks 100 \
  --target-accuracy 0.85 \
  --checkpoint-dir data/artifacts/outbox \
  --state-out data/artifacts/outbox/final_state.h5 \
  --metrics-out data/artifacts/training/metrics.jsonl
```

После запуска training-run проверяй:

1. `data/artifacts/training/metrics.jsonl` (epoch + robustness записи)
2. `data/artifacts/outbox/epoch_XXXXXXXXX.h5` (checkpoint каждой эпохи)
3. `data/artifacts/outbox/final_state.h5` (финальное состояние)

## Пример с параметрами

```bash
docs/step16-acceptance/scripts/run_acceptance.sh \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

## Полезные флаги

- `--skip-build`
- `--skip-lint`
- `--skip-sanitize`
- `--skip-docker`
- `--skip-training`
- `--skip-ws-sparsity`
- `--dataset mnist|synthetic`
- `--max-active-ratio`, `--max-prune-drop`, `--max-noise-drop`
- `--metrics-path <path>`
- `--state-path <path>`
- `--config <path>`, `--checkpoint-dir <path>`, `--data-root <path>`
- `PYTHON_BIN=python3.12 .../run_acceptance.sh` (если нужен другой Python)

## Соответствие DoD пунктам

1. Pipeline MNIST -> class: `check_inference_pipeline.py`
2. Точность >85%: `check_dod_metrics.py` (`eval_accuracy`)
3. Разреженность <5%: `check_ws_sparsity.py` + `active_ratio` из метрик
4. remove_neurons(0.1) loss <5%: `check_dod_metrics.py`
5. inject_noise(0.3) loss <10%: `check_dod_metrics.py`
6. Интерференционные картины (визуально + корреляция): ручная валидация
7. Grafana dashboards: health-check + ручной просмотр
8. 3D visualizer: health-check + ручной просмотр
9. Docker Compose одной командой: `make up`
10. CI зелёный: проверка в GitHub Actions
11. Детерминизм: запуск train 2 раза с одним seed и сравнение артефактов
12. HDF5 воспроизводимость: покрывается `make test` (`test_persistence`)
13. clang-tidy без warning: `make lint`
14. ASan/UBSan clean: sanitize-этап

## Важно про пункт 6

Количественная корреляция картин по классам (0-9) требует отдельного экспорта
воксельных карт активности по меткам. В текущем runbook автоматизирована только
инфраструктура и базовые DoD-гейты; пункт 6 закрывается вручную/дополнительным
экспортом.

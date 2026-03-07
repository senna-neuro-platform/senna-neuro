# Acceptance Runbook

Цель: пройти проверку DoD MVP от развёртывания до итогового PASS/FAIL.

## Что входит

- Пошаговый сценарий запуска.
- Автоматизация ключевых проверок.
- Явные критерии PASS/FAIL.

## Файлы

- `docs/acceptance/scripts/run_acceptance.sh`
- `docs/acceptance/scripts/check_dod_metrics.py`
- `docs/acceptance/scripts/check_inference_pipeline.py`
- `docs/acceptance/scripts/check_ws_sparsity.py`

## Быстрый старт

```bash
cd senna-neuro
chmod +x docs/acceptance/scripts/run_acceptance.sh
docs/acceptance/scripts/run_acceptance.sh
```

Скрипт выполнит:

1. `make install`, `make build-release`, `ctest --preset release`
2. `make lint`
3. `make build-sanitize` + `ctest --preset sanitize`
4. `make up` + health-check endpoints
5. паузу на ручное открытие Grafana/Visualizer и затем полный train-run на реальном MNIST 60k/10k без synthetic fallback
6. проверку метрик DoD (`accuracy`, `robustness`, `max_active_ratio`)
7. проверку pipeline inference (`MNIST image -> class 0..9`)
8. проверку sparsity по WebSocket (`activeCount/totalNeurons < 0.05`) на реальном visualizer trace

## Предварительные условия

Перед шагами 15 и 16 должны быть выполнены:

1. `python3`, `conan`, `cmake`, `ninja`, `clang-tidy`, `docker compose`, `ruff`, `pytest`
2. доступен `g++` с `libasan.so`
3. в `data/MNIST/raw` лежат 4 файла MNIST (`make install` скачивает их автоматически)
4. установлен `torchvision` в текущем Python env
5. локально открыт доступ к портам `3000`, `8000`, `8080`

Базовая подготовка:

```bash
make install
make build-release
ctest --preset release
```

## Памятки Наблюдения Между Шагами

1. После `make up`: проверь состояние контейнеров через `docker compose ps`, логи через `make logs`, и что открываются `http://localhost:3000`, `http://localhost:8080/health`, `http://localhost:8000/health`.
2. Во время training-run: смотри `tail -f data/artifacts/training/metrics.jsonl`; параллельно в Grafana открой `SENNA Training` и `SENNA Activity`.
3. После начала training-run проверь `data/artifacts/metrics/latest.json` и `http://localhost:8000/metrics`: exporter должен отдавать реальные метрики только после появления свежего snapshot.
4. После начала training-run проверь `data/artifacts/visualizer/latest.json` и `http://localhost:8080/lattice`: visualizer должен читать только реальный trace и до его появления честно ждать данные.
5. После проверки DoD-метрик: сверяй `eval_accuracy`, `senna_max_active_neurons_ratio`, `prune_drop`, `noise_drop` в JSONL с графиками Grafana, чтобы подтвердить совпадение телеметрии.
6. После WS-проверки sparsity: открой `http://localhost:8080`, включи heatmap и покадровый режим `Next Tick`, визуально проверь волну и разреженность на реальном trace.

## Как Запускать Training-Run

Вариант A: через orchestrator (рекомендуется)

```bash
docs/acceptance/scripts/run_acceptance.sh \
  --skip-build \
  --skip-lint \
  --skip-sanitize \
  --skip-docker \
  --skip-ws-sparsity \
  --dataset mnist \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

Вариант B: напрямую `python/train.py`

```bash
make install
make build-release
PYTHONPATH=build/release:python python3 python/train.py \
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
  --metrics-out data/artifacts/training/metrics.jsonl \
  --metrics-snapshot-path data/artifacts/metrics/latest.json \
  --visualizer-trace-path data/artifacts/visualizer/latest.json
```

После запуска training-run проверяй:

1. `data/artifacts/training/metrics.jsonl` (epoch + robustness записи)
2. `data/artifacts/outbox/epoch_XXXXXXXXX.h5` (checkpoint каждой эпохи)
3. `data/artifacts/outbox/final_state.h5` (финальное состояние)
4. `data/artifacts/metrics/latest.json` (реальный snapshot для exporter/Grafana)
5. `data/artifacts/visualizer/latest.json` (реальный lattice + per-tick trace для visualizer/WebSocket)

## Закрытие Шага 15

Шаг 15 закрывается только после полного training-run на реальном MNIST и фиксации артефактов.

1. Подготовь данные и release-сборку:

```bash
make install
make build-release
ctest --preset release
```

2. Запусти baseline training-run:

```bash
PYTHONPATH=build/release:python python3 python/train.py \
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
  --metrics-out data/artifacts/training/metrics.jsonl \
  --metrics-snapshot-path data/artifacts/metrics/latest.json \
  --visualizer-trace-path data/artifacts/visualizer/latest.json
```

3. По ходу run наблюдай:
   - `tail -f data/artifacts/training/metrics.jsonl`
   - `ls data/artifacts/outbox/epoch_*.h5`
   - `cat data/artifacts/metrics/latest.json`
   - `cat data/artifacts/visualizer/latest.json`
   - `curl -fsS http://localhost:8000/metrics`
   - `curl -fsS http://localhost:8080/lattice`

4. После завершения зафиксируй evidence:
   - `data/artifacts/training/metrics.jsonl`
   - `data/artifacts/outbox/final_state.h5`
   - минимум один `epoch_XXXXXXXXX.h5`
   - `data/artifacts/metrics/latest.json`
   - `data/artifacts/visualizer/latest.json`

5. Проверь DoD пункты шага 15:

```bash
python3 docs/acceptance/scripts/check_dod_metrics.py \
  --metrics-path data/artifacts/training/metrics.jsonl \
  --require-dataset mnist \
  --target-accuracy 0.85 \
  --max-active-ratio 0.05 \
  --max-prune-drop 0.05 \
  --max-noise-drop 0.10

python3 docs/acceptance/scripts/check_inference_pipeline.py \
  --state-path data/artifacts/outbox/final_state.h5 \
  --data-root data \
  --dataset mnist
```

Шаг 15 считаем закрытым, если:

1. inference pipeline возвращает класс `0..9`
2. `eval_accuracy >= 0.85`
3. `senna_max_active_neurons_ratio <= 0.05`
4. `prune_drop <= 0.05`
5. `noise_drop <= 0.10`
6. epoch checkpoints, `final_state.h5`, `data/artifacts/metrics/latest.json` и `data/artifacts/visualizer/latest.json` созданы без ошибок

## Закрытие Шага 16

Шаг 16 закрывает эксплуатационные и quality-gate требования поверх шага 15.

1. Полный автоматический прогон:

```bash
docs/acceptance/scripts/run_acceptance.sh \
  --epochs 5 \
  --train-limit 60000 \
  --test-limit 10000 \
  --target-accuracy 0.85 \
  --ticks 100
```

Скрипт остановится после `make up` и `print_observe_stack_memo`, чтобы можно было открыть Grafana и visualizer в браузере.
Для полностью неинтерактивного прогона используйте `--no-observe-pause`.

2. Подними runtime отдельно, если нужен ручной осмотр:

```bash
make up
docker compose ps
curl -fsS http://localhost:3000/api/health
curl -fsS http://localhost:8080/health
curl -fsS http://localhost:8000/health
```

3. После training-run проверь exporter:
   - `cat data/artifacts/metrics/latest.json`
   - `curl -fsS http://localhost:8000/metrics`
   - до появления snapshot exporter не должен отдавать synthetic/искусственные метрики

4. После training-run проверь visualizer trace:
   - `cat data/artifacts/visualizer/latest.json`
   - `curl -fsS http://localhost:8080/lattice`
   - до появления trace visualizer не должен подменять lattice или websocket synthetic-данными

5. Проверь Grafana:
   - `http://localhost:3000`
   - дашборды `SENNA Training`, `SENNA Activity`, `SENNA Performance`
   - метрики `senna_test_accuracy`, `senna_active_neurons_ratio`, `senna_spikes_per_tick`

6. Проверь visualizer:
   - `http://localhost:8080`
   - режим `Next Tick`
   - heatmap
   - фильтрацию по типам нейронов

7. Проверь WebSocket sparsity:

```bash
python3 docs/acceptance/scripts/check_ws_sparsity.py \
  --ws-url ws://localhost:8080/ws \
  --max-ratio 0.05
```

8. Для пункта 6 DoD вручную собери evidence по интерференционным картинам:
   - screen/video visualizer для нескольких классов
   - метрики/correlation report, если делается внешний анализ
   - фиксируй, что наблюдаемый паттерн не вырождается в равномерный шум

Шаг 16 считаем закрытым, если:

1. acceptance orchestration проходит без FAIL
2. Docker stack поднимается одной командой `make up`
3. Grafana, exporter и visualizer доступны, exporter не подменяет данные synthetic fallback, visualizer не подменяет lattice/frames synthetic trace
4. WebSocket-проверка sparsity проходит
5. все quality gates 13 и 14 зелёные

## Пример с параметрами

```bash
docs/acceptance/scripts/run_acceptance.sh \
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
- `--no-observe-pause`
- `--dataset mnist`
- `--max-active-ratio`, `--max-prune-drop`, `--max-noise-drop`
- `--metrics-path <path>`
- `--metrics-snapshot-path <path>`
- `--visualizer-trace-path <path>`
- `--state-path <path>`
- `--config <path>`, `--checkpoint-dir <path>`, `--data-root <path>`
- `PYTHON_BIN=python3.12 .../run_acceptance.sh` (если нужен другой Python)

## DoD 13: clang-tidy clean

Команда закрытия:

```bash
make lint
```

Что именно проверяется:

1. `clang-format` для `src/` и `tests/`
2. `scripts/run_clang_tidy.py --build-dir build/debug`
3. `ruff check .`

Важно:

1. `clang-tidy` идёт не по одному `src/main.cpp`, а по всем `.cpp` из `build/debug/compile_commands.json`
2. любой warning трактуется как ошибка через `--warnings-as-errors=*`
3. тот же full-project прогон выполняется в GitHub Actions

## DoD 14: ASan/UBSan clean

Команда закрытия:

```bash
make build-sanitize
ctest --preset sanitize
```

Что важно понимать:

1. sanitize-прогон включает C++ GTest/CTest и Python integration test
2. для pybind11-модуля `senna_core` `CTest` автоматически прокидывает `LD_PRELOAD=<libasan.so>`
3. `ASAN_OPTIONS=detect_leaks=0` задан намеренно: CPython как хост-процесс даёт leak-шум, который не относится к утечкам внутри `senna_core`
4. `UBSAN_OPTIONS=print_stacktrace=1` оставлен для диагностики UB-регрессий

Если `ctest --preset sanitize` зелёный, шаг 14 закрыт.

## Соответствие DoD пунктам

1. Pipeline MNIST -> class: `check_inference_pipeline.py`
2. Точность >85%: `check_dod_metrics.py` (`eval_accuracy`)
3. Разреженность <5%: `check_ws_sparsity.py` + `senna_max_active_neurons_ratio` по всем `epoch_end`
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
14. ASan/UBSan clean: `make build-sanitize` + `ctest --preset sanitize`

## Важно про пункт 6

Количественная корреляция картин по классам (0-9) требует отдельного экспорта
воксельных карт активности по меткам. В текущем runbook автоматизирована только
инфраструктура и базовые DoD-гейты; пункт 6 закрывается вручную/дополнительным
экспортом.

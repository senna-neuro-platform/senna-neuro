# Changelog

## 07.03.2026 `0.16.2-dev`
- В `docker-compose.yml` старт MinIO сделан устойчивым: добавлен HTTP healthcheck, `minio-init` ждёт `service_healthy` и `mc ready`, а `artifact-uploader` больше не стартует по гонке раньше MinIO.
- В `docs/acceptance/scripts/run_acceptance.sh` добавлен ранний preflight для реального MNIST: проверяются host Python-модули `torch` и `torchvision`, наличие `data/MNIST/raw/*`, health endpoint MinIO и явная памятка, что MinIO используется только для артефактов, а не для dataset input.
- В `python/senna/training.py`, `python/train.py` и `python/tests/test_integration.py` уточнён контракт ошибок для MNIST: вместо размытого сообщения training-run теперь отдельно сообщает про отсутствие `torch`/`torchvision` или локальных raw-файлов, и это покрыто pytest-тестами.
- В `docs/acceptance/README.md`, `README.md` и ADR-0011 синхронизирована документация: реальный MNIST читается локально из `data/MNIST/raw`, для него нужны `torch` и `torchvision`, а MinIO хранит только epoch/state артефакты.

## 07.03.2026 `0.16.1-dev`
- В `docs/acceptance/scripts/run_acceptance.sh` acceptance-runtime переведён на release-сборку: вместо `build/debug` и `make test` используются `make build-release`, `ctest --preset release` и `PYTHONPATH=build/release:python` для training/inference-проверок.
- В `docs/acceptance/scripts/run_acceptance.sh` добавлена обязательная пауза после `print_observe_stack_memo`: скрипт ждёт, пока оператор откроет Grafana и visualizer, и продолжает training-run только после ввода `continue`; для неинтерактивного режима добавлен флаг `--no-observe-pause`.
- В `docs/acceptance/README.md` обновлены пошаговые инструкции шага 16 под release-runtime и интерактивную паузу наблюдения перед training-run.

## 07.03.2026 `0.16.0-dev`
- В `src/bindings/python_module.cpp`, `python/senna/training.py` и `python/train.py` добавлен экспорт реального visualizer trace: pybind отдаёт lattice и per-tick activity, а training-run пишет `data/artifacts/visualizer/latest.json` вместе с checkpoint и metrics snapshot.
- В `visualizer/server.js`, `visualizer/index.html` и `docker-compose.yml` visualizer переведён на чтение только реального trace из `data/artifacts`; до появления артефакта `/lattice` не подменяется synthetic-данными, UI честно ждёт trace, а Docker-монтирование даёт доступ к общему artifact volume.
- В `python/senna/training.py` исправлена семантика robustness-gates: `prune_pass` и `noise_pass` больше не могут проходить при нулевой baseline accuracy.
- В `docs/acceptance/scripts/run_acceptance.sh`, `docs/acceptance/README.md`, `README.md` и ADR-0006 зафиксирован новый acceptance-контракт: exporter и visualizer используют только реальные артефакты, orchestration чистит stale trace и ждёт готовность `/metrics` и `/lattice` после training-run.

## 07.03.2026 `0.15.6-dev`
- В `python/train.py` отключён неявный fallback на synthetic при `--dataset mnist`; training-run шага 15 теперь требует реальный MNIST и пишет свежий exporter snapshot в `data/artifacts/metrics/latest.json`.
- В `infra/simulator/simulator_server.py` удалён synthetic fallback метрик: exporter читает только реальный snapshot, `/metrics` отдаёт `503` до его появления, а `/health` явно сообщает `snapshot_ready`.
- В `src/core/metrics/metrics_collector.h` добавлен `senna_max_active_neurons_ratio`, а `docs/acceptance/scripts/check_dod_metrics.py` переведён на проверку максимальной разреженности по всем `epoch_end`, с обязательным `dataset_mode=mnist`.
- В `docs/acceptance/scripts/run_acceptance.sh` и `docs/acceptance/README.md` синхронизированы шаги 15/16: acceptance теперь работает только с `mnist`, очищает stale exporter snapshot, проверяет реальный `/metrics`, использует корректные имена Grafana-метрик и CLI-флаги WebSocket-check.
- Обновлены `README.md`, ADR-0010 и тесты exporter/metrics под новый контракт без synthetic метрик.

## 07.03.2026 `0.15.5-dev`
- В `docs/acceptance/README.md` убран абсолютный локальный путь к workspace; команды запуска приведены к нейтральному виду без данных о локальном окружении.
- В `.clang-tidy` зафиксирован прагматичный full-project профиль проверок (`clang-analyzer`, `bugprone`, `performance` и точечные `modernize`/`readability`) для `src/` и `tests/`, пригодный для стабильного quality gate без мусорных warning.
- Добавлен `scripts/run_clang_tidy.py`: последовательный прогон `clang-tidy` по всем translation units из `build/debug/compile_commands.json` с `--warnings-as-errors=*`.
- В `Makefile` цель `make lint` переведена на полный прогон `clang-format` + `scripts/run_clang_tidy.py` + `ruff check`, вместо проверки одного `src/main.cpp`.
- В `CMakeLists.txt` закрыт sanitize-прогон Python integration test: при `SENNA_ENABLE_SANITIZERS=ON` для `pytest` автоматически выставляются `LD_PRELOAD=<libasan.so>`, `ASAN_OPTIONS=detect_leaks=0` и `UBSAN_OPTIONS=print_stacktrace=1`.
- В `.github/workflows/ci.yml` `clang-tidy` переведён на новый full-project runner, а sanitize configure/build/test теперь выполняются и на `push`, и на `pull_request`.
- В `docs/acceptance/README.md` расширены сценарии закрытия шагов 15 и 16: отдельные процедуры для training-run на MNIST, перечень обязательных evidence-артефактов, ручные проверки Grafana/Visualizer и точные команды для DoD 13/14.
- В `docs/acceptance/README.md` добавлен явный раздел `Как запускать training-run` с двумя сценариями: через `run_acceptance.sh` (training-only режим) и напрямую через `python/train.py`.
- В runbook зафиксированы обязательные выходные артефакты training-run: `metrics.jsonl`, `epoch_XXXXXXXXX.h5`, `final_state.h5`.
- В `docs/acceptance/README.md` добавлены промежуточные памятки наблюдения между шагами приёмки: где смотреть Docker-состояние, Grafana-дашборды, exporter и визуализатор.
- В `docs/acceptance/scripts/run_acceptance.sh` добавлены автоматические блоки `Observation memo` после `make up`/health-check и после training-run с командами для live-наблюдения (`docker compose ps`, `make logs`, `tail metrics.jsonl`, probe exporter).
- Добавлен runbook финальной приёмки MVP: `docs/acceptance/README.md` с пошаговым сценарием от развёртывания до DoD-гейтов шага 16.
- Добавлен orchestration-скрипт `docs/acceptance/scripts/run_acceptance.sh` для автоматического прогона build/test/lint/sanitize, docker health-check, train-run и DoD-проверок.
- Добавлен скрипт `docs/acceptance/scripts/check_dod_metrics.py` для валидации числовых DoD-гейтов по `metrics.jsonl` (`accuracy`, `active_ratio`, `prune_drop`, `noise_drop`).
- Добавлен скрипт `docs/acceptance/scripts/check_inference_pipeline.py` для проверки пути `state + sample -> prediction [0..9]` через Python bindings.
- Добавлен скрипт `docs/acceptance/scripts/check_ws_sparsity.py` для проверки разреженности кадров визуализатора по WebSocket (`activeCount/totalNeurons < 5%`) без внешних зависимостей.
- В `src/bindings/python_module.cpp` усилен training-контур шага 15: `supervise()` теперь выполняет детерминированное обновление весов сенсорных входов к правильному/ошибочному выходу с clamp по `stdp.w_max`.
- В биндингах подключено применение `encoder.max_rate` для плотности входных спайков и `decoder.W_wta` для латерального торможения (WTA) через инъекцию inhibitory-событий.
- В `python/senna/training.py` добавлены helper-функции `evaluate_from_state` и `robustness_report` для воспроизводимой оценки сохранённых состояний и робастности.
- `python/train.py` расширен до сценария шага 15: epoch-checkpoints `epoch_XXXXXXXXX.h5`, early-stop по `target_accuracy`, JSONL-лог (`data/artifacts/training/metrics.jsonl`), диагностические подсказки и post-training проверки `remove_neurons(0.1)`/`inject_noise(0.3)`.
- Обновлён `configs/default.yaml` с зафиксированными гиперпараметрами шага 15 (включая `training.target_accuracy`, `training.learning_rate`, `encoder.max_rate`, обновлённый `w_init_range`).
- Расширены Python integration tests (`python/tests/test_integration.py`) smoke-проверкой `robustness_report`.
- Добавлен ADR-0012 `docs/adr/0012-training-target-and-robustness-gates.md` с фиксацией целевых quality gates шага 15.

## 07.03.2026 `0.14.0-dev`
- Добавлен pybind11-модуль `senna_core` в `src/bindings/python_module.cpp` с контрактом шага 14: `create_network`, `load_sample`, `step`, `get_prediction`, `get_metrics`, `save_state`, `load_state`, `inject_noise`, `remove_neurons`, `supervise`.
- В C++ биндингах реализована строгая YAML-валидация `configs/default.yaml` через `yaml-cpp` (обязательные секции и диапазоны параметров), без дублирования в Python-слое.
- Добавлен единый конфиг `configs/default.yaml` со всеми разделами гиперпараметров: `lattice`, `neuron`, `synapse`, `stdp`, `homeostasis`, `structural`, `encoder`, `decoder`, `training`.
- В `CMakeLists.txt` подключена сборка Python-модуля `senna_core` и интеграционный запуск `python/tests/test_integration.py` через `pytest` в CTest.
- Добавлен Python training pipeline: `python/senna/training.py` и `python/train.py` (MNIST через `torchvision` с fallback на synthetic dataset, цикл `load_sample -> step -> predict -> supervise`).
- Добавлены Python integration tests `python/tests/test_integration.py` для полного цикла API, supervision и save/load.
- В CI (`.github/workflows/ci.yml`) добавлена установка `pytest` для запуска Python integration-тестов.
- Добавлен ADR-0011 `docs/adr/0011-python-bindings-and-training-contract.md` с фиксацией контракта биндингов и training pipeline.

## 07.03.2026 `0.13.0-dev`
- Реализован WebSocket-сервер визуализатора в `visualizer/server.js`: endpoint `/ws`, поток кадров `{tick, neurons:[...]}` только по активным нейронам, endpoint `/lattice` для полной геометрии решётки и `/health` для проверки сервиса.
- В `visualizer/server.js` добавлена детерминированная генерация 3D-решётки и волнового паттерна активности (интерференционные фронты) с ограничением разреженности активных нейронов (<5% от общего числа на кадр).
- Полностью переработан `visualizer/index.html`: рендер решётки через `Three.js InstancedMesh`, цветовая кодировка типов (E/I/Output), вспышки спайков с затуханием на 3-5 кадров и orbit-камера.
- В интерфейс визуализатора добавлены контролы: pause/resume, покадровый шаг `Next Tick`, слайдер скорости проигрывания, фильтрация по типам нейронов, слайсер по `Z`-слоям и переключение в режим тепловой карты активности.
- Добавлена клиентская очередь кадров WebSocket с автопереподключением, что позволяет стабильный real-time режим и управляемый покадровый просмотр без потери визуальной целостности.
- В `README.md` добавлено описание runtime-визуализатора (HTTP + WebSocket endpoint и доступные режимы/контролы).

## 07.03.2026 `0.12.0-dev`
- Добавлен `MetricsCollector` в `src/core/metrics/metrics_collector.h`: сбор метрик по событиям `SimulationEngine` (спайки/тик), расчёт доли активных нейронов, `spikes_per_tick`, средних частот E/I, `ei_balance`, счётчиков STDP и структурной пластичности.
- В `MetricsCollector` добавлен экспорт снапшота в карту метрик (`as_metric_map`) с Prometheus-совместимыми именами (`senna_*`) для дальнейшей передачи в Python exporter.
- Добавлен GTest `tests/test_metrics.cpp`: проверка корректности метрик после 100 детерминированных тактов и проверка экспортируемых ключей/значений.
- В `CMakeLists.txt` подключён `test_metrics` и добавлен CTest `test_prometheus_exporter_format` для валидации Prometheus-формата Python exporter.
- Переписан `infra/simulator/simulator_server.py` в Prometheus exporter с полным набором метрик шага 12 (`active ratio`, `spikes/tick`, `E/I`, `train/test accuracy`, `synapse count`, `pruned/sprouted`, `tick_duration_seconds` histogram, `stdp_updates_total`).
- Добавлен Python-тест `infra/simulator/test_simulator_server.py` на валидность Prometheus payload и загрузку snapshot-метрик из JSON-файла.
- В `docker-compose.yml` для сервиса `simulator` добавлены `METRICS_SNAPSHOT_PATH` и volume `./data/artifacts:/artifacts` для автоматического чтения runtime-снапшотов метрик.
- Добавлены три провиженных Grafana dashboard JSON: `SENNA Activity`, `SENNA Training`, `SENNA Performance`; удалён placeholder-дашборд.
- В `README.md` добавлена документация по метрикам/дашбордам и endpoint exporter `http://localhost:8000/metrics`.

## 07.03.2026 `0.11.3-dev`
- В `EpochArtifactPipeline` (`src/core/persistence/epoch_artifact_pipeline.h`) формат outbox-файла эпохи расширен до 9 цифр: `data/artifacts/outbox/epoch_XXXXXXXXX.h5`.
- В `tests/test_persistence.cpp` добавлена явная проверка имени outbox-файла (`epoch_000000002.h5`) для фиксации нового формата.
- В `README.md` обновлены примеры и описание outbox-пути под 9-значный индекс эпохи.
- Добавлен `EpochArtifactPipeline` в `src/core/persistence/epoch_artifact_pipeline.h`: одним вызовом пишет данные эпохи в основной experiment HDF5 и автоматически формирует outbox-файл `data/artifacts/outbox/epoch_XXXXXX.h5` для фонового uploader.
- В `EpochArtifactPipeline` реализован атомарный паттерн записи outbox-файла (`.tmp` -> rename), чтобы uploader не подхватывал частично записанные epoch-артефакты.
- Добавлена возможность сохранять в outbox и снимок состояния (`/state`) через интеграцию со `StateSerializer`, чтобы восстановление было доступно прямо из epoch-файла.
- В `tests/test_persistence.cpp` добавлен тест `EpochArtifactPipelineTest.WritesEpochFileToOutboxAutomatically` с проверкой генерации outbox-файла и корректного чтения сохраненного `/state`.
- В `README.md` добавлена документация по автоматическому формированию epoch-файлов из C++ persistence (`EpochArtifactPipeline`) для работы MinIO uploader без ручных шагов.
- В `docker-compose.yml` добавлены сервисы `minio`, `minio-init` и `artifact-uploader` для S3-совместимого хранения артефактов и фоновой выгрузки.
- Добавлен контейнерный uploader (`infra/artifact-uploader/Dockerfile`, `infra/artifact-uploader/uploader.py`) с политикой batched-upload: порог по числу эпох (`UPLOAD_BATCH_EPOCHS`) + принудительный flush по таймеру (`UPLOAD_FLUSH_INTERVAL_SEC`), с ограничением размера батча.
- Для uploader добавлена конфигурация `configs/storage/artifact_uploader.env` (S3 endpoint/credentials/bucket/prefix и параметры пакетной фоновой отправки).
- В `README.md` добавлены endpoints MinIO и описание потока артефактов через `data/artifacts/outbox` с фоновым батч-выгрузчиком.
- Добавлен модуль Persistence в `src/core/persistence/hdf5_writer.h`: запись/чтение `spike_trace`, `snapshot` (нейроны+синапсы) и `metrics` в HDF5 с группировкой по эпохам.
- Добавлен `StateSerializer` в `src/core/persistence/state_serializer.h`: сохранение/загрузка полного состояния симуляции (нейроны, синапсы, pending events, `elapsed`, `dt`, `rng_state`) и восстановление runtime-структур.
- В `Neuron` (`src/core/domain/neuron.h`) добавлен сериализуемый снимок `NeuronSnapshot` и API `snapshot()/from_snapshot()/restore_from_snapshot()` для round-trip восстановления состояния.
- В `EventQueue` (`src/core/engine/event_queue.h`) добавлены `snapshot()/restore()` для сериализации очереди отложенных событий.
- Добавлены GTest-тесты `tests/test_persistence.cpp`: bitwise round-trip `spike_trace`, round-trip снапшота, round-trip метрик и проверка детерминированного продолжения симуляции после `save/load`.
- Тест `test_persistence` подключен в `CMakeLists.txt` с линковкой `HDF5::HDF5` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.10.0-dev`
- Реализован `StructuralPlasticity` в `src/core/plasticity/structural_plasticity.h`: прунинг слабых связей по порогу `w_min`, спрутинг новых связей для тихих нейронов (`r_avg < r_target * quiet_ratio`) и периодический запуск раз в `N` тактов.
- В `StructuralPlasticity` добавлен цикл `prune + sprout + rebuild_indices` с метриками `pruned/sprouted` за шаг и накопительными счетчиками.
- При спрутинге используется `Lattice::neighbors` в заданном радиусе, фильтрация уже существующих связей и создание новых синапсов с весом `sprout_weight`.
- Добавлены GTest-тесты `tests/test_structural_plasticity.cpp`: удаление слабого синапса, сохранение сильного, появление новых входов у тихого нейрона, стабильность количества связей после prune+sprout и проверка запуска по интервалу `N` тактов.
- Тест `test_structural_plasticity` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.9.0-dev`
- Реализован `Homeostasis` в `src/core/plasticity/homeostasis.h`: EMA-оценка `r_avg`, корректировка порога к `r_target`, clamp в диапазоне `[theta_min, theta_max]`, обновление по окну `N` тактов.
- В `Neuron` (`src/core/domain/neuron.h`) добавлены методы управления гомеостазом: `set_average_rate`, `set_threshold`, `adjust_threshold`.
- В `SimulationEngine` расширен Observer API: поддержаны подписки на спайки и завершение такта (`set_*`/`add_*` observers), что позволяет подключать медленные контуры обучения без зашивания логики в `tick()`.
- В `Network` (`src/core/engine/network_builder.h`) добавлены прокси-методы для observer-подписок и non-const доступ к `neurons`/`synapses` для правил пластичности.
- Добавлены GTest-тесты `tests/test_homeostasis.cpp`: рост порога у гиперактивного нейрона, снижение порога у молчащего, соблюдение `theta`-границ и сходимость `r_avg` к целевой частоте при длительном прогоне.
- Тест `test_homeostasis` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.8.0-dev`
- Добавлен интерфейс пластичности `IPlasticityRule` в `src/core/plasticity/iplasticity_rule.h` с событиями `on_pre_spike` и `on_post_spike`.
- Реализован `STDPRule` в `src/core/plasticity/stdp.h`: каузальное/антикаузальное обновление весов по экспоненциальному окну, мягкое ограничение потенцирования и жесткий clamp по `w_max`.
- Добавлен `Supervisor` в `src/core/plasticity/supervisor.h` для формирования корректирующего teacher-spike на правильный выходной нейрон при ошибке классификации.
- В `SimulationEngine` добавлен observer-хук на спайк (`set_spike_observer`) для интеграции пластичности через событийную подписку.
- В `SynapseStore` добавлен non-const доступ к синапсам (`at`/`synapses`) для модификации весов правилами пластичности.
- Добавлены GTest-тесты `tests/test_stdp.cpp`: каузальная и антикаузальная пары, затухание эффекта при большом `delta_t`, ограничение `w_max` и рост веса к правильному выходу при супервизии.
- Тест `test_stdp` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.7.2-dev`
- В `Makefile` добавлен таргет `data-mnist`: загрузка MNIST в `data/MNIST/raw` (идемпотентно, с проверкой уже существующих файлов), и подключение этого шага в `make install`.
- В `.gitignore` добавлен `data/` для хранения локально скачанного датасета вне git.
- Переименованы заголовочные файлы IO-интерфейсов без символа `_`: `src/core/io/i_encoder.h -> src/core/io/iencoder.h` и `src/core/io/i_decoder.h -> src/core/io/idecoder.h`; обновлены include-ссылки.
- Добавлены IO-интерфейсы `IEncoder` и `IDecoder` в `src/core/io/iencoder.h` и `src/core/io/idecoder.h`.
- Реализован `RateEncoder` в `src/core/io/rate_encoder.h`: кодирование `MNIST 28x28` в поток `SpikeEvent` по правилу `rate = pixel/255 * max_rate` и вероятности спайка `rate * dt / 1000`.
- Реализован `FirstSpikeDecoder` в `src/core/io/first_spike_decoder.h`: декодирование по первому выходному спайку и генерация латерального торможения (WTA) для остальных выходных нейронов.
- В `SimulationEngine` добавлен экспорт спайков текущего такта (`emitted_events_last_tick`) для сбора выходной активности в end-to-end пайплайне.
- В `Network` добавлен доступ к спайкам последнего такта (`emitted_spikes_last_tick`) для интеграции с декодером.
- Добавлены GTest-тесты `tests/test_io.cpp`: проверка `RateEncoder` (black/medium/white), `FirstSpikeDecoder` (первый спайк, tie-break, WTA) и сквозного пути `encode -> simulate -> decode`.
- Тест `test_io` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.6.0-dev`
- Реализован `NetworkBuilder` и агрегирующий `Network` в `src/core/engine/network_builder.h`: сборка `Lattice -> SynapseStore -> EventQueue -> TimeManager -> SimulationEngine` с детерминируемым seed.
- В `Network` добавлены методы `inject_spike(NeuronId, Time)`, `tick()` и `simulate(duration_ms)` для первого сквозного прогона волны через сеть.
- В `SimulationEngine` добавлен счетчик `emitted_last_tick` для фиксации числа сгенерированных спайков на такте.
- В `Lattice` добавлен non-const доступ к вектору нейронов для интеграции с `SimulationEngine` внутри агрегирующего `Network`.
- Добавлены интеграционные GTest-тесты `tests/test_network_builder.cpp`: тишина без стимула, распространение волны от одного стимула, сравнение `1 vs 10` стимулов и детерминированность трассы.
- Тест `test_network_builder` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.5.0-dev`
- Реализован `Engine: EventQueue` в `src/core/engine/event_queue.h`: очередь событий на `std::priority_queue` с минимальным `arrival` наверху, методы `push` и `drain_tick([t_start, t_end))`.
- Реализован `Engine: TimeManager` в `src/core/engine/time_manager.h`: хранение виртуального времени, шага `dt` (по умолчанию `0.5 ms`), методы `advance`, `elapsed`, `reset`.
- Реализован `Engine: SimulationEngine` в `src/core/engine/simulation_engine.h`: `tick()` доставляет события нейронам, обрабатывает спайки и планирует новые события по исходящим синапсам (`arrival = spike_time + delay`, `value = weight * sign`).
- Добавлены GTest-тесты `tests/test_event_queue.cpp`: порядок извлечения по времени, квантование по интервалу такта, цепочка распространения `A→B→C` с задержками и проверка пустого такта.
- Тест `test_event_queue` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.4.0-dev`
- Реализован `Domain: Lattice` в `src/core/domain/lattice.h`: конфиг решетки, хранение вокселей (`NeuronId` или пусто), плоский массив `Neuron` и `NeighborInfo`.
- Добавлена детерминируемая генерация решетки: сенсорный слой `Z=0` заполняется полностью, обрабатывающий объем `Z=1..D-2` заполняется по плотности, выходной слой `Z=D-1` содержит ровно 10 нейронов с равномерным распределением.
- Зафиксированы правила типов нейронов при размещении: сенсорный и выходной слои полностью `Excitatory`, в объеме используется распределение `80/20` (`Excitatory`/`Inhibitory`).
- Добавлен поиск соседей `neighbors(NeuronId, radius)` наивным перебором куба и предвычисление соседей для базового радиуса в CSR-формате (`offsets + data`).
- Добавлены GTest-тесты `tests/test_lattice.cpp`: размеры/плотность, корректность слоев, проверка соседей (центр и угол) и детерминированность генерации.
- Тест `test_lattice` подключен в `CMakeLists.txt` и зарегистрирован в `CTest` через `gtest_discover_tests`.

## 07.03.2026 `0.3.2-dev`
- `Makefile` переведен на проектные Conan-профили (`build/conan/profiles/host|build`) с автогенерацией по версии локального `g++`, что убирает warnings от `conan profile detect`.
- В CI (`.github/workflows/ci.yml`) добавлен шаг подготовки тех же Conan-профилей и условное добавление `conancenter` remote, чтобы не получать warning `Remote ... already exists`.
- Для Conan-команд в `Makefile` и GitHub Actions добавлено подавление известных upstream deprecated-warning (`core:skip_warnings=["deprecated"]`) от рецепта `hdf5`.
- Реализован `Domain: Synapse` в `src/core/domain/synapse.h`: `Synapse` (`pre_id`, `post_id`, `weight`, `delay`, `sign`) и `SynapseStore` с хранением в плоском массиве.
- В `SynapseStore` добавлены индексы `outgoing`/`incoming`, методы `add`, `connect`, `connect_random`, `rebuild_indices`, а также вычисление `delay = distance * c_base` и знака по типу пресинаптического нейрона.
- Добавлен набор GTest-тестов `tests/test_synapse.cpp`: задержка по расстоянию, знак E/I, диапазон случайного веса, корректность индексов и проверка масштаба `~300k` синапсов.
- Тест `test_synapse` подключен в CMake и запускается через `ctest`.

## 07.03.2026 `0.2.2-dev`
- Зафиксирован ADR-0009: стандарт C++ тестов на GoogleTest и регистрация кейсов в CTest через `gtest_discover_tests`.
- Индекс ADR обновлен: добавлен ADR-0009.
- Тесты `tests/test_types.cpp` и `tests/test_neuron.cpp` переведены на GoogleTest (`TEST`, `ASSERT_*`, `EXPECT_*`) вместо ручных проверок и `main`.
- В `CMakeLists.txt` тесты подключены через `find_package(GTest)` и регистрируются в `CTest` через `gtest_discover_tests`, каждый кейс виден как отдельный тест.
- Реализован `Domain: Neuron` в `src/core/domain/neuron.h`: состояние LIF-нейрона, параметры (`V_rest`, `V_reset`, `tau_m`, `t_ref`, `theta_base`) и метод `receive_input(Time, Weight) -> std::optional<SpikeEvent>`.
- В `receive_input` добавлены аналитическое затухание мембранного потенциала, проверка рефрактерного окна, генерация спайка с `reset` состояния и детерминированное обновление внутреннего состояния.
- Добавлены тесты шага 2 в `tests/test_neuron.cpp`: затухание, срабатывание и reset, рефрактерный период, знак спайка для E/I и детерминированность.
- Тест `test_neuron` подключен в CMake и запускается через `ctest`.

## 07.03.2026 `0.1.4-dev`
- В CI (`.github/workflows/ci.yml`) Conan install теперь запускается с `-s compiler.cppstd=gnu23`.
- В `Makefile` для `install`, `build-release`, `build-sanitize` зафиксирован Conan-флаг `-s compiler.cppstd=gnu23`.
- Зафиксирован ADR-0008 по политике использования шаблонов в C++ ядре.
- В `Domain` добавлено обоснованное применение templates: обобщенный `Coord3<T>` (с alias `Coord3D`) и `ArrivalEarlier<EventT>`.
- Добавлены тесты для шаблонных сценариев (`Coord3<uint16_t>`/`Coord3<int>` и шаблонный компаратор событий).
- Зафиксирован базовый набор проектных ADR-решений (архитектура, стек, модель симуляции, границы MVP, наблюдаемость, quality gates).
- Добавлен индекс ADR: `docs/adr/README.md`.
- Зафиксирован ADR-0001 с правилами версионирования (`A.B.C-dev`) и ведения changelog.
- Реализован Шаг 1 Domain: добавлены базовые типы (`NeuronId`, `SynapseId`, `Time`, `Voltage`, `Weight`), `NeuronType`, `Coord3D::distance()` и `SpikeEvent` в `src/core/domain/types.h`.
- Добавлен юнит-тест `test_types` с проверками расстояния, порядка `SpikeEvent` и различия типов нейронов; тест подключен в `ctest`.

## 07.03.2026 `0.0.5-dev`
- Базовая инфраструктура проекта: CMake/Ninja, Conan, CI, Docker Compose.
- Заглушки simulator/prometheus/grafana/visualizer и стартовый README bootstrap.
- Добавлен `Makefile` с командами: `install`, `lint`, `build-debug`, `build-release`, `build-sanitize`, `test`.
- Добавлены команды управления контейнерами: `make up`, `make down`, `make logs`.
- В README добавлен блок с быстрыми `make`-командами.
- Устранен конфликт дублирующихся CMake preset'ов при повторных вызовах `make build-*` через Conan.
- Команды `build-debug`, `build-release`, `build-sanitize`, `test`, `lint` теперь запускаются стабильно в одной и той же рабочей копии.
- Для Conan-команд в `Makefile` добавлено отключение генерации `CMakeUserPresets.json`, чтобы избежать конфликта preset'ов при повторных сборках.

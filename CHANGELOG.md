# Changelog

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

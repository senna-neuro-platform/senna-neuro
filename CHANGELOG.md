# Changelog

## 07.03.2026 `0.1.3-dev`
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
- Добавлен `Makefile` с командами: `install`, `lint`, `build-debug`, `build-release`, `build-sanitize`, `build-saniryze`, `test`.
- Добавлены команды управления контейнерами: `make up`, `make down`, `make logs`.
- В README добавлен блок с быстрыми `make`-командами.
- Устранен конфликт дублирующихся CMake preset'ов при повторных вызовах `make build-*` через Conan.
- Команды `build-debug`, `build-release`, `build-sanitize`, `build-saniryze`, `test`, `lint` теперь запускаются стабильно в одной и той же рабочей копии.
- Для Conan-команд в `Makefile` добавлено отключение генерации `CMakeUserPresets.json`, чтобы избежать конфликта preset'ов при повторных сборках.

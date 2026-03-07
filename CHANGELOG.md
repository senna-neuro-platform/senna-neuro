# Changelog

Формат версий: `A.B.C-dev`
- `A` — мажорная версия (до первого релиза всегда `0`)
- `B` — номер шага MVP из плана реализации
- `C` — номер сборки/исправления внутри текущего шага (каждое изменение кода +1)

## 07.03.2026 `0.0.5-dev`
- Базовая инфраструктура проекта: CMake/Ninja, Conan, CI, Docker Compose.
- Заглушки simulator/prometheus/grafana/visualizer и стартовый README bootstrap.
- Добавлен `Makefile` с командами: `install`, `lint`, `build-debug`, `build-release`, `build-sanitize`, `build-saniryze`, `test`.
- Добавлены команды управления контейнерами: `make up`, `make down`, `make logs`.
- В README добавлен блок с быстрыми `make`-командами.
- Устранен конфликт дублирующихся CMake preset'ов при повторных вызовах `make build-*` через Conan.
- Команды `build-debug`, `build-release`, `build-sanitize`, `build-saniryze`, `test`, `lint` теперь запускаются стабильно в одной и той же рабочей копии.
- Для Conan-команд в `Makefile` добавлено отключение генерации `CMakeUserPresets.json`, чтобы избежать конфликта preset'ов при повторных сборках.

# senna-neuro

Spatial-Event Neuromorphic Network Architecture.

## Versioning

- Единый источник версии: `VERSION`
- Формат: `A.B.C-dev`
- `A`: до релиза всегда `0`
- `B`: номер шага MVP из плана (при реализации нового шага увеличение на +1, а значение С сбрасывается на 0)
- `C`: каждое изменение кода внутри шага увеличивает на `+1`
- Краткий журнал изменений: `CHANGELOG.md` (группировка по шагу `B`)

## Dev commands

Make shortcuts:

```bash
make install
make lint
make build-debug
make build-release
make build-sanityze
make test
make up
make down
make logs
```

Build presets:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

```bash
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
```

Conan dependencies:

```bash
conan profile detect --force
conan install . --output-folder=build/conan-debug --build=missing -s build_type=Debug
```

Observability stack:

```bash
docker compose up -d
docker compose down
```

## Endpoints

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Visualizer: http://localhost:8080

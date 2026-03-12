# Changelog

## `0.21.1-dev`
- Reset the repository for the next-generation prototype: removed the prior MVP runtime, Python training pipeline, acceptance runbooks/tests, and vendored visualizer assets to start from a minimal scaffold.
- In [CMakeLists.txt](CMakeLists.txt) and [CMakePresets.json](CMakePresets.json), rebuilt the build system around C++23 with Ninja presets for debug/release/sanitize, optional ASan/UBSan via `SENNA_ENABLE_SANITIZERS`, and distinct `senna_core`/`senna_trainer` executables linked through a shared `senna_core_lib` target.
- In [Makefile](Makefile), rewrote the developer workflow: `configure-*` presets drive Conan + CMake, `build-*`/`test-*` mirror the presets, `fmt`/`tidy` cover C++ sources, and docker helpers build/push versioned core/trainer/visualizer images and manage the compose stack.
- Added CI on GitHub Actions ([.github/workflows/ci.yml](.github/workflows/ci.yml)) that installs Conan, runs `make lint`, builds the debug preset, and executes the CTest suite on Ubuntu.
- Introduced a new container stack in [docker-compose.yml](docker-compose.yml) with dedicated Dockerfiles for core, trainer, and visualizer; Prometheus/Grafana provisioning now ships default scrape config and empty placeholder dashboards in [configs/prometheus/prometheus.yml](configs/prometheus/prometheus.yml) and [configs/grafana/*](configs/grafana).
- Implemented a stub core runtime in [src/core/main.cpp](src/core/main.cpp) that opens TCP listeners on gRPC (50051), WebSocket (8080), and metrics (9090) ports, replying `200 OK` on the metrics socket and shutting down cleanly on SIGINT/SIGTERM.
- Added a trainer bootstrap in [src/trainer/main.cpp](src/trainer/main.cpp) that waits for `SENNA_CORE_HOST:SENNA_CORE_PORT` to accept connections before idling, enabling compose-start sequencing.
- Implemented a deterministic voxel lattice generator in [src/core/spatial/lattice.hpp](src/core/spatial/lattice.hpp) and [src/core/spatial/lattice.cpp](src/core/spatial/lattice.cpp): fills a 3D grid by density/seed, tracks neuron IDs, and exposes coordinate lookups.
- Replaced the heavy visualizer with a minimal WebSocket status stub in [visualizer/index.html](visualizer/index.html), [visualizer/app.js](visualizer/app.js), and [visualizer/package.json](visualizer/package.json) served via `http-server` on port 8081.
- Updated Conan metadata in [conanfile.py](conanfile.py) to declare the new dependency set (Boost 1.88, gRPC 1.78, spdlog 1.14, fmt 10.2, yaml-cpp 0.8, GTest 1.17) and generate CMake toolchains/layout.

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential cmake ninja-build python3-pip git \
    && PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-cache-dir "conan>=2.5,<3" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN conan profile detect --force \
 && conan install . --output-folder=build --build=missing -s build_type=Release \
 && cmake -B build -S . \
      -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target senna_trainer -j"$(nproc)"

FROM ubuntu:24.04

WORKDIR /app
COPY --from=builder /src/build/senna_trainer /usr/local/bin/senna_trainer

CMD ["/usr/local/bin/senna_trainer"]

FROM ubuntu:24.04 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target senna_trainer

FROM ubuntu:24.04

WORKDIR /app
COPY --from=builder /src/build/senna_trainer /usr/local/bin/senna_trainer

CMD ["/usr/local/bin/senna_trainer"]

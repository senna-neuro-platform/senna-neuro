#pragma once

#include <string>

#include "core/encoding/rate_encoder.hpp"
#include "core/network/network_builder.hpp"
#include "core/plasticity/structural.hpp"
#include "core/temporal/time_manager.hpp"

namespace senna::config {

struct RuntimeConfig {
  network::NetworkConfig network{};
  struct Ports {
    int grpc = 50051;
    int ws = 8080;
    int metrics = 9090;
  } ports;
  int loop_sleep_ms = 1;
  struct Observability {
    std::vector<double> tick_duration_buckets{0.0005, 0.001, 0.002, 0.005,
                                              0.01,   0.02,  0.05,  0.1};
    int exporter_backlog = 8;
  } observability;
  struct Decoder {
    uint64_t seed = 42;
  } decoder_cfg;
  struct Trainer {
    std::string host = "senna-core";
    int port = 50051;
  } trainer;
};

// Load configuration from YAML file. If the file is missing or a field is
// absent, defaults are used.
RuntimeConfig LoadRuntimeConfig(const std::string& path);

}  // namespace senna::config

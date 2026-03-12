#pragma once

#include <string>

#include "core/encoding/rate_encoder.hpp"
#include "core/network/network_builder.hpp"
#include "core/temporal/time_manager.hpp"

namespace senna::config {

struct RuntimeConfig {
  network::NetworkConfig network{};
};

// Load configuration from YAML file. If the file is missing or a field is
// absent, defaults are used.
RuntimeConfig LoadRuntimeConfig(const std::string& path);

}  // namespace senna::config

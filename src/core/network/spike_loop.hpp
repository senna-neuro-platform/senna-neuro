#pragma once

#include <cstdint>
#include <vector>

#include "core/network/network_builder.hpp"

namespace senna::network {

// Statistics for a single run.
struct RunStats {
  int total_spikes = 0;
  int active_neurons = 0;  // number of distinct neurons that fired
  int ticks = 0;
  float duration_ms = 0.0f;
};

// Main computational loop: runs the network for a given duration.
class SpikeLoop {
 public:
  explicit SpikeLoop(Network& net);

  // Run the network for the given duration (ms).
  // Returns per-run statistics.
  RunStats Run(float duration_ms);

  // Access the full spike log (neuron_id, time) from the last run.
  const std::vector<std::pair<int32_t, float>>& spike_log() const {
    return spike_log_;
  }

 private:
  Network& net_;
  std::vector<std::pair<int32_t, float>> spike_log_;
};

}  // namespace senna::network

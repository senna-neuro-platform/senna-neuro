#pragma once

#include <cstdint>

namespace senna::synaptic {

using SynapseId = int32_t;

// A single synapse connecting two neurons.
struct Synapse {
  int32_t pre_id;   // presynaptic neuron
  int32_t post_id;  // postsynaptic neuron
  float weight;     // synaptic weight (absolute value)
  float delay;      // axonal delay (ms)
  float sign;       // +1.0 (excitatory pre) or -1.0 (inhibitory pre)

  // Effective contribution delivered to post neuron.
  float Effective() const { return weight * sign; }
};

// Parameters for synapse initialization.
struct SynapseParams {
  float w_min = 0.01f;  // minimum initial weight
  float w_max = 0.1f;   // maximum initial weight
  float c_base = 1.0f;  // delay = distance * c_base (ms/voxel)
  float w_wta = -5.0f;  // WTA inhibitory weight for output layer (doc default)
};

inline constexpr SynapseParams kDefaultSynapseParams{};

}  // namespace senna::synaptic

#pragma once

#include <cstdint>

namespace senna::neural {

// Neuron type: excitatory or inhibitory.
enum class NeuronType : uint8_t {
  Excitatory = 0,
  Inhibitory = 1,
};

// LIF neuron constants.
struct LIFParams {
  float V_rest = 0.0f;
  float V_reset = 0.0f;
  float tau_m = 20.0f;      // membrane time constant (ms)
  float t_ref = 2.0f;       // refractory period (ms)
  float theta_base = 1.0f;  // base firing threshold
};

// Default MVP parameters.
inline constexpr LIFParams kDefaultLIF{};

// Single neuron state (AoS view).
// Used for per-neuron logic; bulk storage lives in NeuronPool (SoA).
struct Neuron {
  int x = 0;
  int y = 0;
  int z = 0;
  NeuronType type = NeuronType::Excitatory;
  float V = 0.0f;         // membrane potential
  float theta = 1.0f;     // firing threshold
  float t_last = 0.0f;    // time of last update
  float t_spike = -2.0f;  // time of last spike (init allows firing at t=0)
  float r_avg = 0.0f;     // average firing rate (for homeostasis)

  // Sign of output: +1 excitatory, -1 inhibitory.
  float sign() const { return type == NeuronType::Excitatory ? 1.0f : -1.0f; }

  // Is the neuron in refractory period at time t?
  bool IsRefractory(float t, float t_ref) const {
    return (t - t_spike) < t_ref;
  }
};

}  // namespace senna::neural

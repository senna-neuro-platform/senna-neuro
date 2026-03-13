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
  float V_rest = 0.0F;
  float V_reset = 0.0F;
  float tau_m = 20.0F;      // membrane time constant (ms)
  float t_ref = 2.0F;       // refractory period (ms)
  float theta_base = 1.0F;  // base firing threshold
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
  float V = 0.0F;         // membrane potential
  float theta = 1.0F;     // firing threshold
  float t_last = 0.0F;    // time of last update
  float t_spike = -2.0F;  // time of last spike (init allows firing at t=0)
  float r_avg = 0.0F;     // average firing rate (for homeostasis)

  // Sign of output: +1 excitatory, -1 inhibitory.
  float sign() const { return type == NeuronType::Excitatory ? 1.0F : -1.0F; }

  // Is the neuron in refractory period at time t?
  bool IsRefractory(float t, float t_ref) const {
    return (t - t_spike) < t_ref;
  }
};

}  // namespace senna::neural

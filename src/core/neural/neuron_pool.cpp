#include "core/neural/neuron_pool.hpp"

#include <cmath>

namespace senna::neural {

NeuronPool::NeuronPool(const spatial::Lattice& lattice, const LIFParams& params,
                       double excitatory_ratio, uint64_t seed)
    : size_(lattice.neuron_count()),
      params_(params),
      V_(size_, params.V_rest),
      theta_(size_, params.theta_base),
      t_last_(size_, 0.0f),
      t_spike_(size_, -params.t_ref),  // allow immediate firing at t=0
      r_avg_(size_, 0.0f),
      type_(size_) {
  // Assign E/I types randomly with the given ratio.
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int i = 0; i < size_; ++i) {
    type_[i] = dist(rng) < excitatory_ratio ? NeuronType::Excitatory
                                            : NeuronType::Inhibitory;
  }
}

bool NeuronPool::ReceiveInput(int id, float t_now, float input) {
  // Refractory — ignore input entirely.
  if (IsRefractory(id, t_now)) return false;

  // Lazy exponential decay: V(t) = V_rest + (V_old - V_rest) * exp(-dt/tau_m)
  float dt = t_now - t_last_[id];
  if (dt > 0.0f) {
    float decay = std::exp(-dt / params_.tau_m);
    V_[id] = params_.V_rest + (V_[id] - params_.V_rest) * decay;
  }

  // Integrate input.
  V_[id] += input;
  t_last_[id] = t_now;

  // Check threshold.
  return V_[id] >= theta_[id];
}

void NeuronPool::Fire(int id, float t_now) {
  V_[id] = params_.V_reset;
  t_spike_[id] = t_now;
  t_last_[id] = t_now;
}

}  // namespace senna::neural

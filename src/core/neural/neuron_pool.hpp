#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include "core/neural/neuron.hpp"
#include "core/spatial/lattice.hpp"

namespace senna::neural {

// Structure-of-Arrays storage for all neurons.
// Each field is stored in a contiguous array indexed by NeuronId.
// This layout is cache-friendly for bulk operations on a single field.
class NeuronPool {
 public:
  // Initializes the pool from a lattice.
  // excitatory_ratio: fraction of neurons that are excitatory (0.8 for MVP).
  // seed: RNG seed for E/I type assignment.
  NeuronPool(const spatial::Lattice& lattice, const LIFParams& params,
             double excitatory_ratio, uint64_t seed);

  // --- Per-neuron field accessors (by NeuronId) ---

  float V(int id) const { return V_[id]; }
  float& V(int id) { return V_[id]; }

  float theta(int id) const { return theta_[id]; }
  float& theta(int id) { return theta_[id]; }

  float t_last(int id) const { return t_last_[id]; }
  float& t_last(int id) { return t_last_[id]; }

  float t_spike(int id) const { return t_spike_[id]; }
  float& t_spike(int id) { return t_spike_[id]; }

  float r_avg(int id) const { return r_avg_[id]; }
  float& r_avg(int id) { return r_avg_[id]; }

  NeuronType type(int id) const { return type_[id]; }

  // --- Bulk array access ---

  std::span<float> V_array() { return V_; }
  std::span<const float> V_array() const { return V_; }

  std::span<float> theta_array() { return theta_; }
  std::span<const float> theta_array() const { return theta_; }

  std::span<float> t_last_array() { return t_last_; }
  std::span<float> t_spike_array() { return t_spike_; }
  std::span<float> r_avg_array() { return r_avg_; }

  std::span<const NeuronType> type_array() const { return type_; }

  // --- Queries ---

  int size() const { return size_; }

  // Returns +1.0f for excitatory, -1.0f for inhibitory.
  float sign(int id) const {
    return type_[id] == NeuronType::Excitatory ? 1.0f : -1.0f;
  }

  // Is the neuron in refractory period at time t?
  bool IsRefractory(int id, float t) const {
    return (t - t_spike_[id]) < params_.t_ref;
  }

  const LIFParams& params() const { return params_; }

  // Homeostasis: update r_avg and adjust thresholds toward target firing rate.
  // fired: list of neuron IDs that fired in the last tick.
  // alpha: smoothing factor for r_avg (close to 1 -> slower).
  // target_rate: desired per-tick probability of firing.
  // theta_step: incremental adjustment applied each tick.
  // global_activity: optional fraction of neurons that fired this tick to
  // blend with local r_avg.
  void ApplyHomeostasis(const std::vector<int32_t>& fired, float alpha,
                        float target_rate, float theta_step,
                        float global_activity = -1.0f);

  // --- LIF dynamics ---

  // Lazy decay + input integration.
  // Analytically decays V from t_last to t_now, then adds input.
  // If refractory, input is ignored.
  // Returns true if the neuron fires (V >= theta after integration).
  bool ReceiveInput(int id, float t_now, float input);

  // Fire the neuron: reset V, record spike time.
  void Fire(int id, float t_now);

  // AoS view: gather all fields of neuron id into a Neuron struct.
  Neuron Get(int id, const spatial::Lattice& lattice) const {
    auto [x, y, z] = lattice.CoordsOf(id);
    return {x,         y,          z,           type_[id],
            V_[id],    theta_[id], t_last_[id], t_spike_[id],
            r_avg_[id]};
  }

  // Scatter a Neuron struct back into SoA arrays (type is immutable).
  void Set(int id, const Neuron& n) {
    V_[id] = n.V;
    theta_[id] = n.theta;
    t_last_[id] = n.t_last;
    t_spike_[id] = n.t_spike;
    r_avg_[id] = n.r_avg;
  }

 private:
  int size_;
  LIFParams params_;

  // SoA arrays - all sized to size_.
  std::vector<float> V_;        // membrane potential
  std::vector<float> theta_;    // firing threshold
  std::vector<float> t_last_;   // time of last update
  std::vector<float> t_spike_;  // time of last spike
  std::vector<float> r_avg_;    // average firing rate (for homeostasis)
  std::vector<NeuronType> type_;
};

}  // namespace senna::neural

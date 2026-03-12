#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include "core/neural/neuron.hpp"
#include "core/spatial/lattice.hpp"

namespace senna::plasticity {
struct HomeostasisConfig;
}

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

  float theta(int id) const { return theta_active()[id]; }

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

  std::span<const float> theta_array() const { return theta_active(); }

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
  // target_rate_hz: desired firing rate (Hz).
  // theta_step: homeostatic learning rate (delta theta per Hz error).
  // dt_ms: simulation step in milliseconds (used to convert r_avg to Hz).
  // global_activity: optional fraction of neurons that fired this tick to
  // blend with local r_avg.
  // Update exponentially smoothed firing rates (r_avg) based on fired set.
  void UpdateAverages(const std::vector<int32_t>& fired, float alpha);

  // Apply a freshly computed theta buffer (double-buffer swap).
  void ApplyThetaBuffer(const std::vector<float>& new_theta);

  // Snapshot helpers for background homeostasis worker.
  std::vector<float> ThetaSnapshot() const { return theta_active(); }
  std::vector<float> RateSnapshot() const { return r_avg_; }

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
    return {x,         y,         z,           type_[id],
            V_[id],    theta(id), t_last_[id], t_spike_[id],
            r_avg_[id]};
  }

  // Scatter a Neuron struct back into SoA arrays (type is immutable).
  void Set(int id, const Neuron& n) {
    V_[id] = n.V;
    // keep both buffers in sync when setting explicitly
    for (auto& buf : theta_bufs_) buf[id] = n.theta;
    t_last_[id] = n.t_last;
    t_spike_[id] = n.t_spike;
    r_avg_[id] = n.r_avg;
  }

 private:
  // Active theta buffer view.
  const std::vector<float>& theta_active() const {
    return theta_bufs_[theta_active_idx_.load(std::memory_order_acquire)];
  }
  std::vector<float>& theta_active() {
    return theta_bufs_[theta_active_idx_.load(std::memory_order_acquire)];
  }

  int size_;
  LIFParams params_;

  // SoA arrays - all sized to size_.
  std::vector<float> V_;  // membrane potential
  // Double-buffered thresholds to allow lock-free swaps from background thread.
  std::array<std::vector<float>, 2> theta_bufs_;
  std::atomic<int> theta_active_idx_{0};
  std::vector<float> t_last_;   // time of last update
  std::vector<float> t_spike_;  // time of last spike
  std::vector<float> r_avg_;    // average firing rate (for homeostasis)
  std::vector<NeuronType> type_;
};

}  // namespace senna::neural

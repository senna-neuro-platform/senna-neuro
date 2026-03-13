#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>
#include <vector>

#include "core/neural/neuron_pool.hpp"
#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"
#include "core/synaptic/synapse.hpp"
#include "core/synaptic/synapse_index.hpp"

namespace senna::plasticity {

struct StructuralConfig {
  float w_min_prune = 0.001F;   // prune if |w| < w_min_prune
  int interval_ticks = 10000;   // how often to run
  float sprout_radius = 2.0F;   // search radius for new presynaptic partners
  float sprout_weight = 0.01F;  // initial weight for new synapses
  float quiet_fraction = 0.5F;  // threshold: r_avg < target * quiet_fraction
};

// Stateless structural plasticity utilities.
class StructuralPlasticity {
 public:
  explicit StructuralPlasticity(StructuralConfig cfg = {}) : cfg_(cfg) {}

  // Perform pruning and sprouting and return a new SynapseIndex copy.
  // - pool: provides types/signs and firing stats (r_avg).
  // - homeo_target_hz: target firing rate (for quiet detection).
  // - syn_params: used for delay scaling (c_base).
  synaptic::SynapseIndex Run(const spatial::Lattice& lattice,
                             const spatial::NeighborIndex& neighbors,
                             const neural::NeuronPool& pool,
                             const synaptic::SynapseIndex& current,
                             float homeo_target_hz,
                             const synaptic::SynapseParams& syn_params) const;

 private:
  StructuralConfig cfg_;
};

// Background worker that periodically applies structural plasticity and
// swaps synapse indices via atomic shared_ptr.
class StructuralWorker {
 public:
  StructuralWorker(const spatial::Lattice& lattice,
                   const spatial::NeighborIndex& neighbors,
                   neural::NeuronPool& pool,
                   std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& store,
                   synaptic::SynapseParams syn_params, StructuralConfig cfg,
                   float homeo_target_hz);
  ~StructuralWorker();

  void Start();
  void Stop();
  void Trigger();
  int interval_ticks() const { return cfg_.interval_ticks; }

  void SetConfig(const StructuralConfig& cfg) { cfg_ = cfg; }

 private:
  void Loop();

  const spatial::Lattice& lattice_;
  const spatial::NeighborIndex& neighbors_;
  neural::NeuronPool& pool_;
  std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& store_;
  synaptic::SynapseParams syn_params_;
  StructuralPlasticity sp_;
  StructuralConfig cfg_;
  float homeo_target_hz_;

  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mutex m_;
  std::condition_variable cv_;
  bool has_task_{false};
};

}  // namespace senna::plasticity

#pragma once

#include <atomic>
#include <memory>
#include <span>

#include "core/encoding/rate_encoder.hpp"
#include "core/neural/neuron.hpp"
#include "core/neural/neuron_pool.hpp"
#include "core/plasticity/homeostasis.hpp"
#include "core/plasticity/stdp.hpp"
#include "core/plasticity/structural.hpp"
#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"
#include "core/synaptic/synapse.hpp"
#include "core/synaptic/synapse_index.hpp"
#include "core/temporal/event_queue.hpp"
#include "core/temporal/time_manager.hpp"

namespace senna::network {

// Configuration for building the network.
struct NetworkConfig {
  int width = 28;
  int height = 28;
  int depth = 20;
  double density = 0.7;
  float neighbor_radius = 2.0f;
  double excitatory_ratio = 0.8;
  int num_outputs = 10;
  float dt = 0.5f;
  uint64_t seed = 42;
  neural::LIFParams lif_params{};
  synaptic::SynapseParams synapse_params{};
  plasticity::HomeostasisConfig homeostasis{};
  plasticity::StructuralConfig structural{};
  encoding::RateEncoderParams encoder_params{};
  float decoder_window_ms = 50.0f;
  plasticity::STDPParams stdp_params{};
};

// Owns all network subsystems and wires them together.
// Acts as mediator - subsystems don't know about each other.
class Network {
 public:
  explicit Network(const NetworkConfig& config);

  // --- Accessors ---
  spatial::ZonedLattice& lattice() { return lattice_; }
  const spatial::ZonedLattice& lattice() const { return lattice_; }

  spatial::NeighborIndex& neighbors() { return neighbors_; }
  neural::NeuronPool& pool() { return pool_; }
  const neural::NeuronPool& pool() const { return pool_; }

  const synaptic::SynapseIndex& synapses() const { return *synapses_cache_; }
  std::shared_ptr<synaptic::SynapseIndex> synapses_ptr() const {
    synapses_cache_ = synapses_ptr_atomic_.load();
    return synapses_cache_;
  }
  std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& synapses_ptr_atomic() {
    return synapses_ptr_atomic_;
  }
  void UpdateSynapses(std::shared_ptr<synaptic::SynapseIndex> idx) {
    synapses_ptr_atomic_.store(std::move(idx));
  }
  plasticity::StructuralWorker* structural_worker() const {
    return structural_worker_.get();
  }

  const std::vector<int32_t>& output_ids() const { return output_ids_; }

  temporal::EventQueue& queue() { return queue_; }
  temporal::TimeManager& time_manager() { return time_manager_; }
  const NetworkConfig& config() const { return config_; }

  // --- Stimulus injection ---

  // Inject a spike event into a specific neuron at a given time.
  void InjectSpike(int32_t neuron_id, float time, float value);

  // Inject spikes into sensory panel neurons at given positions.
  void InjectSensory(int x, int y, float time, float value);

  // Encode a 28x28 image into sensory spikes and enqueue them at t_start.
  void EncodeImage(std::span<const uint8_t> image, float t_start);

 private:
  NetworkConfig config_;
  spatial::ZonedLattice lattice_;
  spatial::NeighborIndex neighbors_;
  neural::NeuronPool pool_;
  std::vector<int32_t> output_ids_;  // must be before synapses_ (init order)
  std::atomic<std::shared_ptr<synaptic::SynapseIndex>> synapses_ptr_atomic_;
  mutable std::shared_ptr<synaptic::SynapseIndex> synapses_cache_;
  temporal::EventQueue queue_;
  encoding::RateEncoder encoder_;
  temporal::TimeManager time_manager_;
  std::unique_ptr<plasticity::StructuralWorker> structural_worker_;
};

}  // namespace senna::network

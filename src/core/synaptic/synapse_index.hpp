#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/neural/neuron_pool.hpp"
#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"
#include "core/synaptic/synapse.hpp"

namespace senna::synaptic {

// Stores all synapses in a flat array with two CSR indices:
//   - by post_id (incoming synapses - for event delivery and STDP)
//   - by pre_id  (outgoing synapses - for spike fan-out)
//
// Construction iterates each neuron's neighbor list and creates one synapse
// per neighbor pair. Weights are uniform-random, delays = distance * c_base.
// Sign is determined by the presynaptic neuron type.
//
// Optionally adds WTA (winner-take-all) inhibitory connections among
// output neurons: each output neuron inhibits all others with weight
// w_wta and zero delay.
//
// Thread-safe for concurrent reads after construction.
class SynapseIndex {
 public:
  // Builds synapses from neighbor lists.
  // output_ids: IDs of output-layer neurons for WTA connections.
  //             Pass empty span to skip WTA.
  SynapseIndex(const spatial::Lattice& lattice,
               const spatial::NeighborIndex& neighbors,
               const neural::NeuronPool& pool,
               std::span<const int32_t> output_ids = {},
               const SynapseParams& params = kDefaultSynapseParams,
               uint64_t seed = 42);

  // --- CSR access: incoming synapses (indexed by post_id) ---

  std::span<const SynapseId> Incoming(int post_id) const {
    return {in_data_.data() + in_offsets_[post_id],
            in_data_.data() + in_offsets_[post_id + 1]};
  }

  int IncomingCount(int post_id) const {
    return static_cast<int>(in_offsets_[post_id + 1] - in_offsets_[post_id]);
  }

  // --- CSR access: outgoing synapses (indexed by pre_id) ---

  std::span<const SynapseId> Outgoing(int pre_id) const {
    return {out_data_.data() + out_offsets_[pre_id],
            out_data_.data() + out_offsets_[pre_id + 1]};
  }

  int OutgoingCount(int pre_id) const {
    return static_cast<int>(out_offsets_[pre_id + 1] - out_offsets_[pre_id]);
  }

  // --- Synapse data access ---

  const Synapse& Get(SynapseId id) const { return synapses_[id]; }
  Synapse& Get(SynapseId id) { return synapses_[id]; }

  int32_t synapse_count() const {
    return static_cast<int32_t>(synapses_.size());
  }
  int32_t wta_count() const { return wta_count_; }

  std::span<const Synapse> synapses() const { return synapses_; }
  std::span<Synapse> synapses() { return synapses_; }

 private:
  void BuildCSR(int neuron_count);

  std::vector<Synapse> synapses_;
  int32_t wta_count_ = 0;

  // CSR index: incoming (by post_id)
  std::vector<uint32_t> in_offsets_;
  std::vector<SynapseId> in_data_;

  // CSR index: outgoing (by pre_id)
  std::vector<uint32_t> out_offsets_;
  std::vector<SynapseId> out_data_;
};

}  // namespace senna::synaptic

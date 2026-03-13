#include "core/synaptic/synapse_index.hpp"

#include <algorithm>
#include <numeric>
#include <random>

namespace senna::synaptic {

SynapseIndex::SynapseIndex(const spatial::Lattice& lattice,
                           const spatial::NeighborIndex& neighbors,
                           const neural::NeuronPool& pool,
                           std::span<const int32_t> output_ids,
                           const SynapseParams& params, uint64_t seed) {
  const int n = pool.size();

  // Phase 1: Create synapses from neighbor lists.
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> w_dist(params.w_min, params.w_max);

  for (int pre = 0; pre < n; ++pre) {
    float pre_sign = pool.sign(pre);
    auto nbrs = neighbors.Neighbors(pre);

    for (const auto& entry : nbrs) {
      int post = entry.id;
      if (post == pre) {
        continue;
      }

      synapses_.push_back({
          .pre_id = pre,
          .post_id = post,
          .weight = w_dist(rng),
          .delay = entry.distance * params.c_base,
          .sign = pre_sign,
      });
    }
  }

  // Phase 2: Add WTA inhibitory connections among output neurons.
  // Each output neuron inhibits all others: weight = |w_wta|, sign = -1, delay
  // = 0.
  for (size_t i = 0; i < output_ids.size(); ++i) {
    for (size_t j = 0; j < output_ids.size(); ++j) {
      if (i == j) {
        continue;
      }
      synapses_.push_back({
          .pre_id = output_ids[i],
          .post_id = output_ids[j],
          .weight = std::abs(params.w_wta),
          .delay = 0.0F,
          .sign = -1.0F,
      });
      ++wta_count_;
    }
  }

  // Phase 3: Build CSR indices.
  BuildCSR(n);
}

SynapseIndex::SynapseIndex(int neuron_count, std::vector<Synapse> synapses,
                           int32_t wta_count)
    : synapses_(std::move(synapses)), wta_count_(wta_count) {
  BuildCSR(neuron_count);
}

void SynapseIndex::BuildCSR(int neuron_count) {
  const int n = neuron_count;
  const auto total = static_cast<SynapseId>(synapses_.size());

  // Outgoing CSR (by pre_id).
  out_offsets_.assign(n + 1, 0);
  for (const auto& s : synapses_) {
    out_offsets_[s.pre_id + 1]++;
  }
  std::partial_sum(out_offsets_.begin(), out_offsets_.end(),
                   out_offsets_.begin());

  out_data_.resize(total);
  std::vector<uint32_t> out_pos(out_offsets_.begin(), out_offsets_.end() - 1);
  for (SynapseId sid = 0; sid < total; ++sid) {
    int pre = synapses_[sid].pre_id;
    out_data_[out_pos[pre]++] = sid;
  }

  // Incoming CSR (by post_id).
  in_offsets_.assign(n + 1, 0);
  for (const auto& s : synapses_) {
    in_offsets_[s.post_id + 1]++;
  }
  std::partial_sum(in_offsets_.begin(), in_offsets_.end(), in_offsets_.begin());

  in_data_.resize(total);
  std::vector<uint32_t> in_pos(in_offsets_.begin(), in_offsets_.end() - 1);
  for (SynapseId sid = 0; sid < total; ++sid) {
    int post = synapses_[sid].post_id;
    in_data_[in_pos[post]++] = sid;
  }
}

}  // namespace senna::synaptic

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/domain/types.h"

namespace senna::core::domain {

struct Synapse {
    NeuronId pre_id{};
    NeuronId post_id{};
    Weight weight{};
    Time delay{};
    std::int8_t sign{1};

    [[nodiscard]] Weight effective_weight() const noexcept {
        return weight * static_cast<Weight>(sign);
    }
};

class SynapseStore final {
   public:
    explicit SynapseStore(std::size_t neuron_count = 0);

    [[nodiscard]] std::size_t neuron_capacity() const noexcept { return outgoing_lists_.size(); }

    [[nodiscard]] std::size_t size() const noexcept { return synapses_.size(); }

    [[nodiscard]] bool empty() const noexcept { return synapses_.empty(); }

    [[nodiscard]] const Synapse& at(SynapseId id) const;

    [[nodiscard]] Synapse& at(SynapseId id);

    [[nodiscard]] const std::vector<Synapse>& synapses() const noexcept { return synapses_; }

    [[nodiscard]] std::vector<Synapse>& synapses() noexcept { return synapses_; }

    [[nodiscard]] const std::vector<SynapseId>& outgoing(NeuronId neuron_id) const noexcept;

    [[nodiscard]] const std::vector<SynapseId>& incoming(NeuronId neuron_id) const noexcept;

    [[nodiscard]] std::span<const SynapseId> outgoing_span(NeuronId neuron_id) const;

    [[nodiscard]] std::span<const SynapseId> incoming_span(NeuronId neuron_id) const;

    SynapseId add(const Synapse& synapse);

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    SynapseId connect(NeuronId pre_id, NeuronId post_id, const Coord3D& pre_position,
                      const Coord3D& post_position, NeuronType pre_type, Weight weight,
                      Time c_base = 1.0F);

    template <typename RandomGenerator>
    SynapseId connect_random(NeuronId pre_id, NeuronId post_id, const Coord3D& pre_position,
                             const Coord3D& post_position, const NeuronType pre_type,
                             RandomGenerator& random, const Time c_base = 1.0F,
                             const Weight min_weight = 0.01F, const Weight max_weight = 0.1F) {
        std::uniform_real_distribution<Weight> dist(min_weight, max_weight);
        return connect(pre_id, post_id, pre_position, post_position, pre_type, dist(random),
                       c_base);
    }

    void rebuild_indices(std::size_t neuron_count = 0);

   private:
    void ensure_neuron_capacity(NeuronId neuron_count);
    void ensure_compact_indices() const;

    std::vector<Synapse> synapses_{};
    std::vector<std::vector<SynapseId>> outgoing_lists_{};
    std::vector<std::vector<SynapseId>> incoming_lists_{};
    mutable std::vector<SynapseId> outgoing_flat_{};
    mutable std::vector<SynapseId> incoming_flat_{};
    mutable std::vector<std::size_t> outgoing_offsets_{};
    mutable std::vector<std::size_t> incoming_offsets_{};
    mutable bool compact_indices_dirty_{true};
};

}  // namespace senna::core::domain

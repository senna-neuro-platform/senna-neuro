#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
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
    explicit SynapseStore(std::size_t neuron_count = 0)
        : outgoing_(neuron_count), incoming_(neuron_count) {}

    [[nodiscard]] std::size_t neuron_capacity() const noexcept { return outgoing_.size(); }

    [[nodiscard]] std::size_t size() const noexcept { return synapses_.size(); }

    [[nodiscard]] bool empty() const noexcept { return synapses_.empty(); }

    [[nodiscard]] const Synapse& at(const SynapseId id) const {
        return synapses_.at(static_cast<std::size_t>(id));
    }

    [[nodiscard]] const std::vector<Synapse>& synapses() const noexcept { return synapses_; }

    [[nodiscard]] const std::vector<SynapseId>& outgoing(const NeuronId neuron_id) const noexcept {
        static const std::vector<SynapseId> empty_index{};
        const auto idx = static_cast<std::size_t>(neuron_id);
        return idx < outgoing_.size() ? outgoing_[idx] : empty_index;
    }

    [[nodiscard]] const std::vector<SynapseId>& incoming(const NeuronId neuron_id) const noexcept {
        static const std::vector<SynapseId> empty_index{};
        const auto idx = static_cast<std::size_t>(neuron_id);
        return idx < incoming_.size() ? incoming_[idx] : empty_index;
    }

    SynapseId add(const Synapse& synapse) {
        ensure_neuron_capacity(std::max(synapse.pre_id, synapse.post_id) + 1U);

        const auto raw_id = synapses_.size();
        if (raw_id > static_cast<std::size_t>(std::numeric_limits<SynapseId>::max())) {
            throw std::overflow_error("SynapseId overflow");
        }

        const auto id = static_cast<SynapseId>(raw_id);
        synapses_.push_back(synapse);
        outgoing_[static_cast<std::size_t>(synapse.pre_id)].push_back(id);
        incoming_[static_cast<std::size_t>(synapse.post_id)].push_back(id);
        return id;
    }

    SynapseId connect(NeuronId pre_id, NeuronId post_id, const Coord3D& pre_position,
                      const Coord3D& post_position, const NeuronType pre_type, const Weight weight,
                      const Time c_base = 1.0F) {
        const auto sign = pre_type == NeuronType::Excitatory ? static_cast<std::int8_t>(1)
                                                             : static_cast<std::int8_t>(-1);
        const auto delay = pre_position.distance(post_position) * c_base;
        return add(Synapse{pre_id, post_id, weight, delay, sign});
    }

    template <typename RandomGenerator>
    SynapseId connect_random(NeuronId pre_id, NeuronId post_id, const Coord3D& pre_position,
                             const Coord3D& post_position, const NeuronType pre_type,
                             RandomGenerator& random, const Time c_base = 1.0F,
                             const Weight min_weight = 0.01F, const Weight max_weight = 0.1F) {
        std::uniform_real_distribution<Weight> dist(min_weight, max_weight);
        return connect(pre_id, post_id, pre_position, post_position, pre_type, dist(random),
                       c_base);
    }

    void rebuild_indices(std::size_t neuron_count = 0) {
        if (neuron_count == 0) {
            for (const auto& synapse : synapses_) {
                neuron_count = std::max(
                    neuron_count,
                    static_cast<std::size_t>(std::max(synapse.pre_id, synapse.post_id) + 1U));
            }
        }

        outgoing_.assign(neuron_count, {});
        incoming_.assign(neuron_count, {});

        for (std::size_t id = 0; id < synapses_.size(); ++id) {
            const auto& synapse = synapses_[id];
            const auto synapse_id = static_cast<SynapseId>(id);
            ensure_neuron_capacity(std::max(synapse.pre_id, synapse.post_id) + 1U);
            outgoing_[static_cast<std::size_t>(synapse.pre_id)].push_back(synapse_id);
            incoming_[static_cast<std::size_t>(synapse.post_id)].push_back(synapse_id);
        }
    }

   private:
    void ensure_neuron_capacity(const NeuronId neuron_count) {
        const auto required = static_cast<std::size_t>(neuron_count);
        if (required <= outgoing_.size()) {
            return;
        }
        outgoing_.resize(required);
        incoming_.resize(required);
    }

    std::vector<Synapse> synapses_{};
    std::vector<std::vector<SynapseId>> outgoing_{};
    std::vector<std::vector<SynapseId>> incoming_{};
};

}  // namespace senna::core::domain

#include "core/domain/synapse.h"

namespace senna::core::domain {

SynapseStore::SynapseStore(const std::size_t neuron_count)
    : outgoing_(neuron_count), incoming_(neuron_count) {}

const Synapse& SynapseStore::at(const SynapseId id) const {
    return synapses_.at(static_cast<std::size_t>(id));
}

Synapse& SynapseStore::at(const SynapseId id) { return synapses_.at(static_cast<std::size_t>(id)); }

const std::vector<SynapseId>& SynapseStore::outgoing(const NeuronId neuron_id) const noexcept {
    static const std::vector<SynapseId> empty_index{};
    const auto idx = static_cast<std::size_t>(neuron_id);
    return idx < outgoing_.size() ? outgoing_[idx] : empty_index;
}

const std::vector<SynapseId>& SynapseStore::incoming(const NeuronId neuron_id) const noexcept {
    static const std::vector<SynapseId> empty_index{};
    const auto idx = static_cast<std::size_t>(neuron_id);
    return idx < incoming_.size() ? incoming_[idx] : empty_index;
}

SynapseId SynapseStore::add(const Synapse& synapse) {
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

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
SynapseId SynapseStore::connect(const NeuronId pre_id, const NeuronId post_id,
                                const Coord3D& pre_position, const Coord3D& post_position,
                                const NeuronType pre_type, const Weight weight, const Time c_base) {
    const auto sign = pre_type == NeuronType::Excitatory ? static_cast<std::int8_t>(1)
                                                         : static_cast<std::int8_t>(-1);
    const auto delay = pre_position.distance(post_position) * c_base;
    return add(Synapse{pre_id, post_id, weight, delay, sign});
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void SynapseStore::rebuild_indices(std::size_t neuron_count) {
    if (neuron_count == 0) {
        for (const auto& synapse : synapses_) {
            neuron_count =
                std::max(neuron_count,
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

void SynapseStore::ensure_neuron_capacity(const NeuronId neuron_count) {
    const auto required = static_cast<std::size_t>(neuron_count);
    if (required <= outgoing_.size()) {
        return;
    }
    outgoing_.resize(required);
    incoming_.resize(required);
}

}  // namespace senna::core::domain

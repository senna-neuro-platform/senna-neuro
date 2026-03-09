#include "core/domain/synapse.h"

namespace senna::core::domain {

SynapseStore::SynapseStore(const std::size_t neuron_count)
    : outgoing_lists_(neuron_count), incoming_lists_(neuron_count) {}

const Synapse& SynapseStore::at(const SynapseId id) const {
    return synapses_.at(static_cast<std::size_t>(id));
}

Synapse& SynapseStore::at(const SynapseId id) { return synapses_.at(static_cast<std::size_t>(id)); }

const std::vector<SynapseId>& SynapseStore::outgoing(const NeuronId neuron_id) const noexcept {
    static const std::vector<SynapseId> empty_index{};
    const auto idx = static_cast<std::size_t>(neuron_id);
    return idx < outgoing_lists_.size() ? outgoing_lists_[idx] : empty_index;
}

const std::vector<SynapseId>& SynapseStore::incoming(const NeuronId neuron_id) const noexcept {
    static const std::vector<SynapseId> empty_index{};
    const auto idx = static_cast<std::size_t>(neuron_id);
    return idx < incoming_lists_.size() ? incoming_lists_[idx] : empty_index;
}

std::span<const SynapseId> SynapseStore::outgoing_span(const NeuronId neuron_id) const {
    ensure_compact_indices();
    const auto idx = static_cast<std::size_t>(neuron_id);
    if (idx + 1U >= outgoing_offsets_.size()) {
        return {};
    }

    const auto begin = outgoing_offsets_[idx];
    const auto end = outgoing_offsets_[idx + 1U];
    return std::span<const SynapseId>(outgoing_flat_.data() + begin, end - begin);
}

std::span<const SynapseId> SynapseStore::incoming_span(const NeuronId neuron_id) const {
    ensure_compact_indices();
    const auto idx = static_cast<std::size_t>(neuron_id);
    if (idx + 1U >= incoming_offsets_.size()) {
        return {};
    }

    const auto begin = incoming_offsets_[idx];
    const auto end = incoming_offsets_[idx + 1U];
    return std::span<const SynapseId>(incoming_flat_.data() + begin, end - begin);
}

SynapseId SynapseStore::add(const Synapse& synapse) {
    ensure_neuron_capacity(std::max(synapse.pre_id, synapse.post_id) + 1U);

    const auto raw_id = synapses_.size();
    if (raw_id > static_cast<std::size_t>(std::numeric_limits<SynapseId>::max())) {
        throw std::overflow_error("SynapseId overflow");
    }

    const auto id = static_cast<SynapseId>(raw_id);
    synapses_.push_back(synapse);
    outgoing_lists_[static_cast<std::size_t>(synapse.pre_id)].push_back(id);
    incoming_lists_[static_cast<std::size_t>(synapse.post_id)].push_back(id);
    compact_indices_dirty_ = true;
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

    outgoing_lists_.assign(neuron_count, {});
    incoming_lists_.assign(neuron_count, {});

    std::vector<std::size_t> outgoing_counts(neuron_count, 0U);
    std::vector<std::size_t> incoming_counts(neuron_count, 0U);
    for (const auto& synapse : synapses_) {
        ++outgoing_counts[static_cast<std::size_t>(synapse.pre_id)];
        ++incoming_counts[static_cast<std::size_t>(synapse.post_id)];
    }

    for (std::size_t neuron = 0; neuron < neuron_count; ++neuron) {
        outgoing_lists_[neuron].reserve(outgoing_counts[neuron]);
        incoming_lists_[neuron].reserve(incoming_counts[neuron]);
    }

    for (std::size_t id = 0; id < synapses_.size(); ++id) {
        const auto& synapse = synapses_[id];
        const auto synapse_id = static_cast<SynapseId>(id);
        ensure_neuron_capacity(std::max(synapse.pre_id, synapse.post_id) + 1U);
        outgoing_lists_[static_cast<std::size_t>(synapse.pre_id)].push_back(synapse_id);
        incoming_lists_[static_cast<std::size_t>(synapse.post_id)].push_back(synapse_id);
    }

    compact_indices_dirty_ = true;
}

void SynapseStore::ensure_neuron_capacity(const NeuronId neuron_count) {
    const auto required = static_cast<std::size_t>(neuron_count);
    if (required <= outgoing_lists_.size()) {
        return;
    }
    outgoing_lists_.resize(required);
    incoming_lists_.resize(required);
    compact_indices_dirty_ = true;
}

void SynapseStore::ensure_compact_indices() const {
    if (!compact_indices_dirty_) {
        return;
    }

    outgoing_offsets_.assign(outgoing_lists_.size() + 1U, 0U);
    incoming_offsets_.assign(incoming_lists_.size() + 1U, 0U);
    outgoing_flat_.clear();
    incoming_flat_.clear();
    outgoing_flat_.reserve(synapses_.size());
    incoming_flat_.reserve(synapses_.size());

    for (std::size_t neuron = 0; neuron < outgoing_lists_.size(); ++neuron) {
        outgoing_offsets_[neuron] = outgoing_flat_.size();
        outgoing_flat_.insert(outgoing_flat_.end(), outgoing_lists_[neuron].begin(),
                              outgoing_lists_[neuron].end());
    }
    outgoing_offsets_[outgoing_lists_.size()] = outgoing_flat_.size();

    for (std::size_t neuron = 0; neuron < incoming_lists_.size(); ++neuron) {
        incoming_offsets_[neuron] = incoming_flat_.size();
        incoming_flat_.insert(incoming_flat_.end(), incoming_lists_[neuron].begin(),
                              incoming_lists_[neuron].end());
    }
    incoming_offsets_[incoming_lists_.size()] = incoming_flat_.size();

    compact_indices_dirty_ = false;
}

}  // namespace senna::core::domain

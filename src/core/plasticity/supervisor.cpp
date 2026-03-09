#include "core/plasticity/supervisor.h"

namespace senna::core::plasticity {

Supervisor::Supervisor(const SupervisorConfig config) : config_(config) {}

std::optional<senna::core::domain::SpikeEvent> Supervisor::correction_event(
    const int predicted_class, const int expected_class,
    const std::vector<senna::core::domain::NeuronId>& output_neurons,
    const senna::core::domain::Time t_now) const {
    if (predicted_class == expected_class) {
        return std::nullopt;
    }
    if (expected_class < 0 || static_cast<std::size_t>(expected_class) >= output_neurons.size()) {
        throw std::out_of_range("Expected class is out of output neuron range");
    }

    const auto neuron_id = output_neurons[static_cast<std::size_t>(expected_class)];
    return senna::core::domain::SpikeEvent{
        neuron_id,
        neuron_id,
        t_now,
        config_.correction_spike_value,
    };
}

std::size_t Supervisor::apply_output_weight_update(
    const int predicted_class, const int expected_class,
    const std::span<const senna::core::domain::NeuronId> output_neurons,
    const std::span<const std::uint16_t> pre_spike_counts,
    senna::core::domain::SynapseStore& synapses, const senna::core::domain::Weight learning_rate,
    const senna::core::domain::Weight min_weight,
    const senna::core::domain::Weight max_weight) const {
    if (expected_class < 0 || static_cast<std::size_t>(expected_class) >= output_neurons.size()) {
        throw std::out_of_range("Expected class is out of output neuron range");
    }
    if (learning_rate <= 0.0F) {
        throw std::invalid_argument("learning_rate must be positive");
    }
    if (min_weight < 0.0F || max_weight < min_weight) {
        throw std::invalid_argument("weight bounds are invalid");
    }

    const auto expected_output = output_neurons[static_cast<std::size_t>(expected_class)];
    const auto has_predicted = predicted_class >= 0 &&
                               static_cast<std::size_t>(predicted_class) < output_neurons.size() &&
                               predicted_class != expected_class;
    const auto predicted_output =
        has_predicted ? output_neurons[static_cast<std::size_t>(predicted_class)] : expected_output;

    std::size_t updated = 0U;
    updated += adjust_output_weights(expected_output, true, pre_spike_counts, synapses,
                                     learning_rate, min_weight, max_weight);
    if (has_predicted) {
        updated += adjust_output_weights(predicted_output, false, pre_spike_counts, synapses,
                                         learning_rate, min_weight, max_weight);
    }
    return updated;
}

senna::core::domain::Weight Supervisor::activity_scale(const std::uint16_t spike_count) noexcept {
    constexpr auto kMaxScaledSpikes = static_cast<std::uint16_t>(8U);
    const auto bounded = std::min(spike_count, kMaxScaledSpikes);
    return static_cast<senna::core::domain::Weight>(bounded) /
           static_cast<senna::core::domain::Weight>(kMaxScaledSpikes);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
std::size_t Supervisor::adjust_output_weights(const senna::core::domain::NeuronId output_id,
                                              const bool reinforce,
                                              const std::span<const std::uint16_t> pre_spike_counts,
                                              senna::core::domain::SynapseStore& synapses,
                                              const senna::core::domain::Weight learning_rate,
                                              const senna::core::domain::Weight min_weight,
                                              const senna::core::domain::Weight max_weight) const {
    std::size_t updated = 0U;

    for (const auto synapse_id : synapses.incoming(output_id)) {
        auto& synapse = synapses.at(synapse_id);
        const auto pre_index = static_cast<std::size_t>(synapse.pre_id);
        if (pre_index >= pre_spike_counts.size()) {
            continue;
        }

        const auto activity = activity_scale(pre_spike_counts[pre_index]);
        if (activity <= 0.0F) {
            continue;
        }

        const auto delta = learning_rate * activity;
        const auto excitatory = synapse.sign >= 0;
        const auto signed_delta =
            reinforce ? (excitatory ? delta : -delta) : (excitatory ? -delta : delta);
        synapse.weight = std::clamp(synapse.weight + signed_delta, min_weight, max_weight);
        ++updated;
    }

    return updated;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace senna::core::plasticity

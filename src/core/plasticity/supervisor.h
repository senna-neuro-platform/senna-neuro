#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/domain/synapse.h"
#include "core/domain/types.h"

namespace senna::core::plasticity {

struct SupervisorConfig {
    senna::core::domain::Weight correction_spike_value{1.1F};
};

class Supervisor final {
   public:
    explicit Supervisor(SupervisorConfig config = {});

    [[nodiscard]] std::optional<senna::core::domain::SpikeEvent> correction_event(
        int predicted_class, int expected_class,
        const std::vector<senna::core::domain::NeuronId>& output_neurons,
        senna::core::domain::Time t_now) const;

    [[nodiscard]] std::size_t apply_output_weight_update(
        int predicted_class, int expected_class,
        const std::span<const senna::core::domain::NeuronId> output_neurons,
        const std::span<const std::uint16_t> pre_spike_counts,
        senna::core::domain::SynapseStore& synapses, senna::core::domain::Weight learning_rate,
        senna::core::domain::Weight min_weight, senna::core::domain::Weight max_weight) const;

   private:
    [[nodiscard]] static senna::core::domain::Weight activity_scale(
        std::uint16_t spike_count) noexcept;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] std::size_t adjust_output_weights(
        senna::core::domain::NeuronId output_id, bool reinforce,
        const std::span<const std::uint16_t> pre_spike_counts,
        senna::core::domain::SynapseStore& synapses, senna::core::domain::Weight learning_rate,
        senna::core::domain::Weight min_weight, senna::core::domain::Weight max_weight) const;

    SupervisorConfig config_{};
};

}  // namespace senna::core::plasticity

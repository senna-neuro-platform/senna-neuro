#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

#include "core/domain/types.h"

namespace senna::core::plasticity {

struct SupervisorConfig {
    senna::core::domain::Weight correction_spike_value{1.1F};
};

class Supervisor final {
   public:
    explicit Supervisor(const SupervisorConfig config = {}) : config_(config) {}

    [[nodiscard]] std::optional<senna::core::domain::SpikeEvent> correction_event(
        const int predicted_class, const int expected_class,
        const std::vector<senna::core::domain::NeuronId>& output_neurons,
        const senna::core::domain::Time t_now) const {
        if (predicted_class == expected_class) {
            return std::nullopt;
        }
        if (expected_class < 0 ||
            static_cast<std::size_t>(expected_class) >= output_neurons.size()) {
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

   private:
    SupervisorConfig config_{};
};

}  // namespace senna::core::plasticity

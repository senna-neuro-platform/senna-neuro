#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/engine/event_queue.h"
#include "core/engine/time_manager.h"

namespace senna::core::engine {

class SimulationEngine final {
   public:
    SimulationEngine(std::vector<senna::core::domain::Neuron>& neurons,
                     const senna::core::domain::SynapseStore& synapses, EventQueue& queue,
                     TimeManager& time) noexcept
        : neurons_(neurons), synapses_(synapses), queue_(queue), time_(time) {}

    void inject_event(const senna::core::domain::SpikeEvent event) { queue_.push(event); }

    [[nodiscard]] std::size_t tick() {
        const auto t_start = time_.elapsed();
        const auto t_end = t_start + time_.dt();
        const auto events = queue_.drain_tick(t_start, t_end);
        emitted_last_tick_ = 0U;

        for (const auto& event : events) {
            const auto target = static_cast<std::size_t>(event.target);
            if (target >= neurons_.size()) {
                throw std::out_of_range("Event target neuron is out of range");
            }

            const auto maybe_spike = neurons_[target].receive_input(event.arrival, event.value);
            if (!maybe_spike.has_value()) {
                continue;
            }
            ++emitted_last_tick_;

            const auto pre = maybe_spike->source;
            for (const auto synapse_id : synapses_.outgoing(pre)) {
                const auto& synapse = synapses_.at(synapse_id);
                queue_.push(senna::core::domain::SpikeEvent{
                    pre,
                    synapse.post_id,
                    maybe_spike->arrival + synapse.delay,
                    synapse.effective_weight(),
                });
            }
        }

        time_.advance();
        return events.size();
    }

    [[nodiscard]] std::size_t emitted_last_tick() const noexcept { return emitted_last_tick_; }

   private:
    std::vector<senna::core::domain::Neuron>& neurons_;
    const senna::core::domain::SynapseStore& synapses_;
    EventQueue& queue_;
    TimeManager& time_;
    std::size_t emitted_last_tick_{0U};
};

}  // namespace senna::core::engine

#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/engine/event_queue.h"
#include "core/engine/time_manager.h"

namespace senna::core::engine {

class SimulationEngine final {
   public:
    using SpikeObserver = std::function<void(const senna::core::domain::SpikeEvent&)>;
    using TickObserver = std::function<void(senna::core::domain::Time, senna::core::domain::Time)>;

    SimulationEngine(std::vector<senna::core::domain::Neuron>& neurons,
                     const senna::core::domain::SynapseStore& synapses, EventQueue& queue,
                     TimeManager& time) noexcept;

    void set_spike_observer(SpikeObserver observer);

    void add_spike_observer(SpikeObserver observer);

    void set_tick_observer(TickObserver observer);

    void add_tick_observer(TickObserver observer);

    void inject_event(senna::core::domain::SpikeEvent event);

    [[nodiscard]] std::size_t tick();

    [[nodiscard]] std::size_t emitted_last_tick() const noexcept { return emitted_last_tick_; }

    [[nodiscard]] const std::vector<senna::core::domain::SpikeEvent>& emitted_events_last_tick()
        const noexcept {
        return emitted_events_last_tick_;
    }

   private:
    std::vector<senna::core::domain::Neuron>& neurons_;
    const senna::core::domain::SynapseStore& synapses_;
    EventQueue& queue_;
    TimeManager& time_;
    std::size_t emitted_last_tick_{0U};
    std::vector<senna::core::domain::SpikeEvent> emitted_events_last_tick_{};
    std::vector<SpikeObserver> spike_observers_{};
    std::vector<TickObserver> tick_observers_{};
};

}  // namespace senna::core::engine

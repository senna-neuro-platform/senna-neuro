#pragma once

#include <cstddef>
#include <cstdint>
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
    void mark_neuron_dirty(std::size_t neuron_index) noexcept;

    void sync_dirty_neurons() noexcept;

    std::vector<senna::core::domain::Neuron>& neurons_;
    const senna::core::domain::SynapseStore& synapses_;
    EventQueue& queue_;
    TimeManager& time_;
    std::vector<senna::core::domain::NeuronId> neuron_ids_{};
    std::vector<senna::core::domain::Voltage> v_rest_{};
    std::vector<senna::core::domain::Voltage> v_reset_{};
    std::vector<senna::core::domain::Time> tau_m_{};
    std::vector<senna::core::domain::Time> t_ref_{};
    std::vector<senna::core::domain::Weight> spike_value_{};
    std::vector<senna::core::domain::Voltage> potential_{};
    std::vector<senna::core::domain::Time> last_update_time_{};
    std::vector<senna::core::domain::Time> last_spike_time_{};
    std::vector<std::uint8_t> in_refractory_{};
    std::vector<std::uint8_t> dirty_bitmap_{};
    std::vector<std::size_t> dirty_neurons_{};
    std::size_t emitted_last_tick_{0U};
    std::vector<senna::core::domain::SpikeEvent> emitted_events_last_tick_{};
    std::vector<SpikeObserver> spike_observers_{};
    std::vector<TickObserver> tick_observers_{};
};

}  // namespace senna::core::engine

#include "core/engine/simulation_engine.h"

namespace senna::core::engine {

SimulationEngine::SimulationEngine(std::vector<senna::core::domain::Neuron>& neurons,
                                   const senna::core::domain::SynapseStore& synapses,
                                   EventQueue& queue, TimeManager& time) noexcept
    : neurons_(neurons), synapses_(synapses), queue_(queue), time_(time) {}

void SimulationEngine::set_spike_observer(SpikeObserver observer) {
    spike_observers_.clear();
    if (observer) {
        spike_observers_.push_back(std::move(observer));
    }
}

void SimulationEngine::add_spike_observer(SpikeObserver observer) {
    if (observer) {
        spike_observers_.push_back(std::move(observer));
    }
}

void SimulationEngine::set_tick_observer(TickObserver observer) {
    tick_observers_.clear();
    if (observer) {
        tick_observers_.push_back(std::move(observer));
    }
}

void SimulationEngine::add_tick_observer(TickObserver observer) {
    if (observer) {
        tick_observers_.push_back(std::move(observer));
    }
}

void SimulationEngine::inject_event(const senna::core::domain::SpikeEvent event) {
    queue_.push(event);
}

std::size_t SimulationEngine::tick() {
    const auto t_start = time_.elapsed();
    const auto t_end = t_start + time_.dt();
    const auto& events = queue_.drain_tick(t_start, t_end);
    emitted_last_tick_ = 0U;
    emitted_events_last_tick_.clear();

    for (const auto& event : events) {
        const auto target = static_cast<std::size_t>(event.target);
        if (target >= neurons_.size()) {
            throw std::out_of_range("Event target neuron is out of range");
        }

        const auto maybe_spike = neurons_[target].receive_input(event.arrival, event.value);
        if (!maybe_spike) {
            continue;
        }
        const auto emitted_spike = *maybe_spike;
        ++emitted_last_tick_;
        emitted_events_last_tick_.push_back(emitted_spike);
        for (auto& observer : spike_observers_) {
            observer(emitted_spike);
        }

        const auto pre = emitted_spike.source;
        for (const auto synapse_id : synapses_.outgoing_span(pre)) {
            const auto& synapse = synapses_.at(synapse_id);
            queue_.push(senna::core::domain::SpikeEvent{
                pre,
                synapse.post_id,
                emitted_spike.arrival + synapse.delay,
                synapse.effective_weight(),
            });
        }
    }

    for (auto& observer : tick_observers_) {
        observer(t_start, t_end);
    }

    time_.advance();
    return events.size();
}

}  // namespace senna::core::engine

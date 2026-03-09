#include "core/engine/simulation_engine.h"

#include <cmath>

namespace senna::core::engine {

SimulationEngine::SimulationEngine(std::vector<senna::core::domain::Neuron>& neurons,
                                   const senna::core::domain::SynapseStore& synapses,
                                   EventQueue& queue, TimeManager& time) noexcept
    : neurons_(neurons),
      synapses_(synapses),
      queue_(queue),
      time_(time),
      neuron_ids_(neurons.size()),
      v_rest_(neurons.size()),
      v_reset_(neurons.size()),
      tau_m_(neurons.size()),
      t_ref_(neurons.size()),
      spike_value_(neurons.size()),
      potential_(neurons.size()),
      last_update_time_(neurons.size()),
      last_spike_time_(neurons.size()),
      in_refractory_(neurons.size(), 0U),
      dirty_bitmap_(neurons.size(), 0U) {
    for (std::size_t index = 0U; index < neurons_.size(); ++index) {
        const auto& neuron = neurons_[index];
        neuron_ids_[index] = neuron.id();
        v_rest_[index] = neuron.config().v_rest;
        v_reset_[index] = neuron.config().v_reset;
        tau_m_[index] = neuron.config().tau_m;
        t_ref_[index] = neuron.config().t_ref;
        spike_value_[index] =
            neuron.type() == senna::core::domain::NeuronType::Excitatory ? 1.0F : -1.0F;
        potential_[index] = neuron.potential();
        last_update_time_[index] = neuron.last_update_time();
        last_spike_time_[index] = neuron.last_spike_time();
        in_refractory_[index] = static_cast<std::uint8_t>(neuron.in_refractory() ? 1U : 0U);
    }
}

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

        auto effective_now = event.arrival;
        if (effective_now < last_update_time_[target]) {
            effective_now = last_update_time_[target];
        }

        in_refractory_[target] = static_cast<std::uint8_t>(
            effective_now < (last_spike_time_[target] + t_ref_[target]) ? 1U : 0U);
        mark_neuron_dirty(target);
        if (in_refractory_[target] != 0U) {
            continue;
        }

        const auto dt = effective_now - last_update_time_[target];
        if (dt > 0.0F) {
            const auto safe_tau_m = tau_m_[target] > 0.0F ? tau_m_[target] : 1.0F;
            const auto decay = std::exp(-(dt / safe_tau_m));
            potential_[target] = v_rest_[target] + ((potential_[target] - v_rest_[target]) * decay);
        }

        potential_[target] += event.value;
        last_update_time_[target] = effective_now;

        if (potential_[target] < neurons_[target].threshold()) {
            continue;
        }

        potential_[target] = v_reset_[target];
        last_spike_time_[target] = effective_now;
        in_refractory_[target] = 1U;

        sync_dirty_neurons();
        const auto emitted_spike = senna::core::domain::SpikeEvent{
            neuron_ids_[target],
            neuron_ids_[target],
            effective_now,
            spike_value_[target],
        };
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

    sync_dirty_neurons();
    for (auto& observer : tick_observers_) {
        observer(t_start, t_end);
    }

    time_.advance();
    return events.size();
}

void SimulationEngine::mark_neuron_dirty(const std::size_t neuron_index) noexcept {
    if (neuron_index >= dirty_bitmap_.size() || dirty_bitmap_[neuron_index] != 0U) {
        return;
    }

    dirty_bitmap_[neuron_index] = 1U;
    dirty_neurons_.push_back(neuron_index);
}

void SimulationEngine::sync_dirty_neurons() noexcept {
    for (const auto neuron_index : dirty_neurons_) {
        neurons_[neuron_index].set_runtime_state(
            potential_[neuron_index], last_update_time_[neuron_index],
            last_spike_time_[neuron_index], in_refractory_[neuron_index] != 0U);
        dirty_bitmap_[neuron_index] = 0U;
    }
    dirty_neurons_.clear();
}

}  // namespace senna::core::engine

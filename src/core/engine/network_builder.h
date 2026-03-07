#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/domain/lattice.h"
#include "core/domain/synapse.h"
#include "core/engine/event_queue.h"
#include "core/engine/simulation_engine.h"
#include "core/engine/time_manager.h"

namespace senna::core::engine {

struct NetworkBuilderConfig {
    senna::core::domain::LatticeConfig lattice{};
    senna::core::domain::Time dt{0.5F};
    senna::core::domain::Time c_base{1.0F};
    senna::core::domain::Weight min_weight{0.01F};
    senna::core::domain::Weight max_weight{0.1F};
    senna::core::domain::Weight input_spike_value{1.1F};
};

class Network final {
   public:
    Network(senna::core::domain::Lattice lattice, senna::core::domain::SynapseStore synapses,
            const senna::core::domain::Time dt, const senna::core::domain::Weight input_spike_value)
        : lattice_(std::move(lattice)),
          synapses_(std::move(synapses)),
          queue_{},
          time_(dt, 0.0F),
          engine_(lattice_.neurons(), synapses_, queue_, time_),
          input_spike_value_(input_spike_value) {}

    Network(const Network&) = delete;
    Network(Network&&) = delete;
    Network& operator=(const Network&) = delete;
    Network& operator=(Network&&) = delete;

    void inject_spike(const senna::core::domain::NeuronId neuron_id,
                      const senna::core::domain::Time arrival) {
        if (static_cast<std::size_t>(neuron_id) >= lattice_.neurons().size()) {
            throw std::out_of_range("Injected neuron id is out of range");
        }

        queue_.push(senna::core::domain::SpikeEvent{
            neuron_id,
            neuron_id,
            arrival,
            input_spike_value_,
        });
    }

    [[nodiscard]] std::size_t tick() {
        static_cast<void>(engine_.tick());
        return engine_.emitted_last_tick();
    }

    [[nodiscard]] const std::vector<senna::core::domain::SpikeEvent>& emitted_spikes_last_tick()
        const noexcept {
        return engine_.emitted_events_last_tick();
    }

    void set_spike_observer(SimulationEngine::SpikeObserver observer) {
        engine_.set_spike_observer(std::move(observer));
    }

    void add_spike_observer(SimulationEngine::SpikeObserver observer) {
        engine_.add_spike_observer(std::move(observer));
    }

    void set_tick_observer(SimulationEngine::TickObserver observer) {
        engine_.set_tick_observer(std::move(observer));
    }

    void add_tick_observer(SimulationEngine::TickObserver observer) {
        engine_.add_tick_observer(std::move(observer));
    }

    [[nodiscard]] std::vector<std::size_t> simulate(const senna::core::domain::Time duration_ms) {
        std::vector<std::size_t> spike_trace{};
        if (duration_ms <= 0.0F) {
            return spike_trace;
        }

        while (time_.elapsed() < duration_ms) {
            spike_trace.push_back(tick());
        }
        return spike_trace;
    }

    [[nodiscard]] senna::core::domain::Time elapsed() const noexcept { return time_.elapsed(); }

    [[nodiscard]] std::size_t synapse_count() const noexcept { return synapses_.size(); }

    [[nodiscard]] const senna::core::domain::Lattice& lattice() const noexcept { return lattice_; }

    [[nodiscard]] const senna::core::domain::SynapseStore& synapses() const noexcept {
        return synapses_;
    }

    [[nodiscard]] senna::core::domain::SynapseStore& synapses() noexcept { return synapses_; }

    [[nodiscard]] std::vector<senna::core::domain::Neuron>& neurons() noexcept {
        return lattice_.neurons();
    }

   private:
    senna::core::domain::Lattice lattice_{};
    senna::core::domain::SynapseStore synapses_{};
    EventQueue queue_{};
    TimeManager time_{};
    SimulationEngine engine_;
    senna::core::domain::Weight input_spike_value_{1.1F};
};

class NetworkBuilder final {
   public:
    explicit NetworkBuilder(const NetworkBuilderConfig config = {}) : config_(config) {
        validate_config();
    }

    [[nodiscard]] Network build(const std::uint32_t seed = 42U) const {
        std::mt19937 random{seed};

        senna::core::domain::Lattice lattice{config_.lattice};
        lattice.generate(random);

        senna::core::domain::SynapseStore synapses{lattice.neuron_count()};
        const auto& neurons = lattice.neurons();
        for (const auto& pre_neuron : neurons) {
            const auto neighbors =
                lattice.neighbors(pre_neuron.id(), config_.lattice.neighbor_radius);
            for (const auto& neighbor : neighbors) {
                const auto& post_neuron = neurons.at(static_cast<std::size_t>(neighbor.id));
                synapses.connect_random(pre_neuron.id(), post_neuron.id(), pre_neuron.position(),
                                        post_neuron.position(), pre_neuron.type(), random,
                                        config_.c_base, config_.min_weight, config_.max_weight);
            }
        }

        return Network{std::move(lattice), std::move(synapses), config_.dt,
                       config_.input_spike_value};
    }

    [[nodiscard]] const NetworkBuilderConfig& config() const noexcept { return config_; }

   private:
    void validate_config() const {
        if (config_.dt <= 0.0F) {
            throw std::invalid_argument("NetworkBuilderConfig.dt must be positive");
        }
        if (config_.c_base <= 0.0F) {
            throw std::invalid_argument("NetworkBuilderConfig.c_base must be positive");
        }
        if (config_.min_weight < 0.0F || config_.max_weight < 0.0F) {
            throw std::invalid_argument("NetworkBuilderConfig weights must be non-negative");
        }
        if (config_.min_weight > config_.max_weight) {
            throw std::invalid_argument("NetworkBuilderConfig min_weight must be <= max_weight");
        }
    }

    NetworkBuilderConfig config_{};
};

}  // namespace senna::core::engine

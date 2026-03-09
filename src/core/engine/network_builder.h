#pragma once

#include <cstdint>
#include <memory>
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
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Network(senna::core::domain::Lattice lattice, senna::core::domain::SynapseStore synapses,
            senna::core::domain::Time dt, senna::core::domain::Weight input_spike_value);

    Network(const Network&) = delete;
    Network(Network&&) = delete;
    Network& operator=(const Network&) = delete;
    Network& operator=(Network&&) = delete;

    void inject_spike(senna::core::domain::NeuronId neuron_id, senna::core::domain::Time arrival);

    void inject_event(const senna::core::domain::SpikeEvent& event);

    void reset_between_samples();

    [[nodiscard]] std::size_t tick();

    [[nodiscard]] const std::vector<senna::core::domain::SpikeEvent>& emitted_spikes_last_tick()
        const noexcept {
        return engine_.emitted_events_last_tick();
    }

    void set_spike_observer(SimulationEngine::SpikeObserver observer);

    void add_spike_observer(SimulationEngine::SpikeObserver observer);

    void set_tick_observer(SimulationEngine::TickObserver observer);

    void add_tick_observer(SimulationEngine::TickObserver observer);

    [[nodiscard]] std::vector<std::size_t> simulate(senna::core::domain::Time duration_ms);

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

    [[nodiscard]] EventQueue& event_queue() noexcept { return queue_; }

    [[nodiscard]] const EventQueue& event_queue() const noexcept { return queue_; }

    [[nodiscard]] TimeManager& time_manager() noexcept { return time_; }

    [[nodiscard]] const TimeManager& time_manager() const noexcept { return time_; }

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
    explicit NetworkBuilder(NetworkBuilderConfig config = {});

    [[nodiscard]] Network build(std::uint32_t seed = 42U) const;

    [[nodiscard]] std::unique_ptr<Network> build_ptr(std::uint32_t seed = 42U) const;

    [[nodiscard]] const NetworkBuilderConfig& config() const noexcept { return config_; }

   private:
    void validate_config() const;

    NetworkBuilderConfig config_{};
};

}  // namespace senna::core::engine

#include "core/engine/network_builder.h"

#include <algorithm>
#include <cmath>

namespace senna::core::engine {

namespace {

std::pair<senna::core::domain::Lattice, senna::core::domain::SynapseStore> build_components(
    const NetworkBuilderConfig& config, const std::uint32_t seed) {
    std::mt19937 random{seed};

    senna::core::domain::Lattice lattice{config.lattice};
    lattice.generate(random);

    senna::core::domain::SynapseStore synapses{lattice.neuron_count()};
    const auto& neurons = lattice.neurons();

    // Local connectivity for all neurons (existing logic)
    for (const auto& pre_neuron : neurons) {
        const auto neighbors = lattice.neighbors(pre_neuron.id(), config.lattice.neighbor_radius);
        for (const auto& neighbor : neighbors) {
            const auto& post_neuron = neurons.at(static_cast<std::size_t>(neighbor.id));
            synapses.connect_random(pre_neuron.id(), post_neuron.id(), pre_neuron.position(),
                                    post_neuron.position(), pre_neuron.type(), random,
                                    config.c_base, config.min_weight, config.max_weight);
        }
    }

    // Global projections: processing volume → output neurons
    // Each output neuron receives input from a random subset of processing neurons
    // so it can integrate information across the entire spatial extent.
    const auto sensor_count = lattice.sensor_neuron_count();
    const auto output_count = lattice.output_neuron_count();
    const auto total_count = neurons.size();
    const auto processing_count = total_count - sensor_count - output_count;

    if (processing_count > 0U && output_count > 0U) {
        // Collect processing neuron indices
        std::vector<std::size_t> processing_indices;
        processing_indices.reserve(processing_count);
        for (std::size_t i = sensor_count; i < total_count - output_count; ++i) {
            processing_indices.push_back(i);
        }

        // Each output neuron gets connections from a random sample of processing neurons.
        // Fan-in: sqrt(processing_count) clamped to [processing_count/10, processing_count].
        const auto fan_in_raw =
            static_cast<std::size_t>(std::sqrt(static_cast<double>(processing_count)));
        const auto fan_in = std::clamp(
            fan_in_raw, std::max<std::size_t>(1U, processing_count / 10U), processing_count);

        for (std::size_t o = 0U; o < output_count; ++o) {
            const auto output_idx = total_count - output_count + o;
            const auto& post_neuron = neurons[output_idx];

            // Shuffle and pick fan_in sources
            auto shuffled = processing_indices;
            for (std::size_t i = shuffled.size() - 1U; i > 0U; --i) {
                std::uniform_int_distribution<std::size_t> dist(0U, i);
                std::swap(shuffled[i], shuffled[dist(random)]);
            }

            const auto count = std::min(fan_in, shuffled.size());
            for (std::size_t j = 0U; j < count; ++j) {
                const auto& pre_neuron = neurons[shuffled[j]];
                synapses.connect_random(pre_neuron.id(), post_neuron.id(), pre_neuron.position(),
                                        post_neuron.position(), pre_neuron.type(), random,
                                        config.c_base, config.min_weight, config.max_weight);
            }
        }
    }

    return {std::move(lattice), std::move(synapses)};
}

}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
Network::Network(senna::core::domain::Lattice lattice, senna::core::domain::SynapseStore synapses,
                 const senna::core::domain::Time dt,
                 const senna::core::domain::Weight input_spike_value)
    : lattice_(std::move(lattice)),
      synapses_(std::move(synapses)),
      queue_{},
      time_(dt, 0.0F),
      engine_(lattice_.neurons(), synapses_, queue_, time_),
      input_spike_value_(input_spike_value) {}
// NOLINTEND(bugprone-easily-swappable-parameters)

void Network::inject_spike(const senna::core::domain::NeuronId neuron_id,
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

void Network::inject_event(const senna::core::domain::SpikeEvent& event) {
    engine_.inject_event(event);
}

void Network::reset_between_samples() {
    queue_.clear();
    time_.reset(0.0F);
    engine_.reset_state();
}

std::size_t Network::tick() {
    static_cast<void>(engine_.tick());
    return engine_.emitted_last_tick();
}

void Network::set_spike_observer(SimulationEngine::SpikeObserver observer) {
    engine_.set_spike_observer(std::move(observer));
}

void Network::add_spike_observer(SimulationEngine::SpikeObserver observer) {
    engine_.add_spike_observer(std::move(observer));
}

void Network::set_tick_observer(SimulationEngine::TickObserver observer) {
    engine_.set_tick_observer(std::move(observer));
}

void Network::add_tick_observer(SimulationEngine::TickObserver observer) {
    engine_.add_tick_observer(std::move(observer));
}

std::vector<std::size_t> Network::simulate(const senna::core::domain::Time duration_ms) {
    std::vector<std::size_t> spike_trace{};
    if (duration_ms <= 0.0F) {
        return spike_trace;
    }

    while (time_.elapsed() < duration_ms) {
        spike_trace.push_back(tick());
    }
    return spike_trace;
}

NetworkBuilder::NetworkBuilder(const NetworkBuilderConfig config) : config_(config) {
    validate_config();
}

Network NetworkBuilder::build(const std::uint32_t seed) const {
    auto [lattice, synapses] = build_components(config_, seed);
    return Network{std::move(lattice), std::move(synapses), config_.dt, config_.input_spike_value};
}

std::unique_ptr<Network> NetworkBuilder::build_ptr(const std::uint32_t seed) const {
    auto [lattice, synapses] = build_components(config_, seed);
    return std::make_unique<Network>(std::move(lattice), std::move(synapses), config_.dt,
                                     config_.input_spike_value);
}

void NetworkBuilder::validate_config() const {
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

}  // namespace senna::core::engine

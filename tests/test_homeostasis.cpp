#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/types.h"
#include "core/engine/event_queue.h"
#include "core/engine/simulation_engine.h"
#include "core/engine/time_manager.h"
#include "core/plasticity/homeostasis.h"
#include "test_support/require_value.h"

TEST(HomeostasisTest, HyperactiveNeuronRaisesThresholdAndReducesSpiking) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::plasticity::Homeostasis;
    using senna::core::plasticity::HomeostasisConfig;

    NeuronConfig neuron_config{};
    neuron_config.theta_base = 0.2F;
    neuron_config.t_ref = 0.0F;
    neuron_config.tau_m = 1e6F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, neuron_config);

    HomeostasisConfig config{};
    config.alpha = 0.99F;
    config.r_target_hz = 5.0F;
    config.eta_homeo = 0.02F;
    config.theta_min = 0.1F;
    config.theta_max = 5.0F;

    Homeostasis homeostasis{config};
    const auto initial_threshold = neurons.front().threshold();

    std::size_t first_half_spikes = 0U;
    std::size_t second_half_spikes = 0U;
    constexpr float dt = 1.0F;

    for (int tick = 0; tick < 400; ++tick) {
        const auto time = static_cast<float>(tick) * dt;
        const auto spike = neurons.front().receive_input(time, 0.3F);
        if (spike.has_value()) {
            homeostasis.on_spike(require_value(spike, "expected spike for hyperactive neuron"));
            if (tick < 200) {
                ++first_half_spikes;
            } else {
                ++second_half_spikes;
            }
        }
        homeostasis.on_tick(neurons, dt);
    }

    EXPECT_GT(neurons.front().threshold(), initial_threshold);
    EXPECT_LT(second_half_spikes, first_half_spikes);
}

TEST(HomeostasisTest, SilentNeuronLowersThreshold) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::plasticity::Homeostasis;
    using senna::core::plasticity::HomeostasisConfig;

    NeuronConfig neuron_config{};
    neuron_config.theta_base = 2.0F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, neuron_config);

    HomeostasisConfig config{};
    config.alpha = 0.99F;
    config.r_target_hz = 5.0F;
    config.eta_homeo = 0.01F;
    config.theta_min = 0.1F;
    config.theta_max = 5.0F;

    Homeostasis homeostasis{config};
    const auto initial_threshold = neurons.front().threshold();

    for (int tick = 0; tick < 500; ++tick) {
        homeostasis.on_tick(neurons, 1.0F);
    }

    EXPECT_LT(neurons.front().threshold(), initial_threshold);
    EXPECT_GE(neurons.front().threshold(), config.theta_min);
}

TEST(HomeostasisTest, ThresholdStaysInsideConfiguredBounds) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::plasticity::Homeostasis;
    using senna::core::plasticity::HomeostasisConfig;

    NeuronConfig neuron_config{};
    neuron_config.theta_base = 0.4F;
    neuron_config.t_ref = 0.0F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, neuron_config);

    HomeostasisConfig config{};
    config.alpha = 0.99F;
    config.r_target_hz = 5.0F;
    config.eta_homeo = 1.0F;
    config.theta_min = 0.3F;
    config.theta_max = 0.6F;

    Homeostasis homeostasis{config};

    for (int tick = 0; tick < 200; ++tick) {
        const auto time = static_cast<float>(tick);
        const auto spike = neurons.front().receive_input(time, 10.0F);
        if (spike.has_value()) {
            homeostasis.on_spike(require_value(spike, "expected spike in bounded-threshold test"));
        }
        homeostasis.on_tick(neurons, 1.0F);
    }

    EXPECT_LE(neurons.front().threshold(), config.theta_max);
    EXPECT_GE(neurons.front().threshold(), config.theta_min);

    for (int tick = 0; tick < 200; ++tick) {
        homeostasis.on_tick(neurons, 1.0F);
    }

    EXPECT_LE(neurons.front().threshold(), config.theta_max);
    EXPECT_GE(neurons.front().threshold(), config.theta_min);
}

TEST(HomeostasisTest, LongRunAverageRateConvergesToTarget) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SpikeEvent;
    using senna::core::plasticity::Homeostasis;
    using senna::core::plasticity::HomeostasisConfig;

    NeuronConfig neuron_config{};
    neuron_config.theta_base = 1.0F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, neuron_config);

    HomeostasisConfig config{};
    config.alpha = 0.999F;
    config.r_target_hz = 5.0F;
    config.eta_homeo = 0.001F;

    Homeostasis homeostasis{config};

    constexpr int total_ticks = 20000;
    constexpr int period_ticks = 200;  // 1 spike every 200 ms => 5 Hz at dt=1 ms

    for (int tick = 0; tick < total_ticks; ++tick) {
        if (tick % period_ticks == 0) {
            homeostasis.on_spike(SpikeEvent{0U, 0U, static_cast<float>(tick), 1.0F});
        }
        homeostasis.on_tick(neurons, 1.0F);
    }

    EXPECT_NEAR(neurons.front().average_rate(), config.r_target_hz, 0.5F);
}

TEST(HomeostasisTest, IntegratesViaSimulationEngineObservers) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SpikeEvent;
    using senna::core::domain::SynapseStore;
    using senna::core::engine::EventQueue;
    using senna::core::engine::SimulationEngine;
    using senna::core::engine::TimeManager;
    using senna::core::plasticity::Homeostasis;
    using senna::core::plasticity::HomeostasisConfig;

    NeuronConfig neuron_config{};
    neuron_config.theta_base = 0.2F;
    neuron_config.t_ref = 0.0F;
    neuron_config.tau_m = 1e6F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, neuron_config);

    SynapseStore synapses{1U};
    EventQueue queue{};
    TimeManager time{1.0F, 0.0F};
    SimulationEngine engine{neurons, synapses, queue, time};

    HomeostasisConfig config{};
    config.alpha = 0.99F;
    config.eta_homeo = 0.02F;
    Homeostasis homeostasis{config};

    engine.add_spike_observer(
        [&homeostasis](const SpikeEvent& spike) { homeostasis.on_spike(spike); });
    engine.add_tick_observer([&homeostasis, &neurons](const float t_start, const float t_end) {
        homeostasis.on_tick(neurons, t_end - t_start);
    });

    const auto initial_threshold = neurons.front().threshold();
    for (int tick = 0; tick < 200; ++tick) {
        const auto t = static_cast<float>(tick);
        engine.inject_event(SpikeEvent{0U, 0U, t, 0.3F});
        static_cast<void>(engine.tick());
    }

    EXPECT_GT(neurons.front().threshold(), initial_threshold);
}

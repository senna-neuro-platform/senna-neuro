#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/domain/types.h"
#include "core/engine/event_queue.h"
#include "core/engine/simulation_engine.h"
#include "core/engine/time_manager.h"

TEST(EventQueueTest, ReturnsEventsInAscendingArrivalOrder) {
    using senna::core::domain::SpikeEvent;
    using senna::core::engine::EventQueue;

    EventQueue queue{};
    queue.push(SpikeEvent{0U, 1U, 2.0F, 0.1F});
    queue.push(SpikeEvent{0U, 1U, 0.5F, 0.1F});
    queue.push(SpikeEvent{0U, 1U, 1.0F, 0.1F});

    const auto drained = queue.drain_tick(0.0F, 3.0F);
    ASSERT_EQ(drained.size(), 3U);
    EXPECT_NEAR(drained[0].arrival, 0.5F, 1e-6F);
    EXPECT_NEAR(drained[1].arrival, 1.0F, 1e-6F);
    EXPECT_NEAR(drained[2].arrival, 2.0F, 1e-6F);
}

TEST(EventQueueTest, DrainsEventsInsideTickInterval) {
    using senna::core::domain::SpikeEvent;
    using senna::core::engine::EventQueue;

    EventQueue queue{};
    queue.push(SpikeEvent{0U, 1U, 1.1F, 0.1F});
    queue.push(SpikeEvent{0U, 1U, 1.3F, 0.1F});
    queue.push(SpikeEvent{0U, 1U, 1.5F, 0.1F});

    const auto first_tick = queue.drain_tick(1.0F, 1.5F);
    ASSERT_EQ(first_tick.size(), 2U);
    EXPECT_NEAR(first_tick[0].arrival, 1.1F, 1e-6F);
    EXPECT_NEAR(first_tick[1].arrival, 1.3F, 1e-6F);

    const auto second_tick = queue.drain_tick(1.5F, 2.0F);
    ASSERT_EQ(second_tick.size(), 1U);
    EXPECT_NEAR(second_tick[0].arrival, 1.5F, 1e-6F);
}

TEST(SimulationEngineTest, PropagatesSpikeAlongAtoBtoCWithDelays) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SpikeEvent;
    using senna::core::domain::SynapseStore;
    using senna::core::engine::EventQueue;
    using senna::core::engine::SimulationEngine;
    using senna::core::engine::TimeManager;

    NeuronConfig config{};
    config.theta_base = 1.0F;
    config.t_ref = 0.0F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, config);
    neurons.emplace_back(1U, Coord3D{1U, 0U, 0U}, NeuronType::Excitatory, config);
    neurons.emplace_back(2U, Coord3D{2U, 0U, 0U}, NeuronType::Excitatory, config);

    SynapseStore synapses{neurons.size()};
    synapses.connect(0U, 1U, neurons[0].position(), neurons[1].position(), neurons[0].type(), 1.1F,
                     1.0F);
    synapses.connect(1U, 2U, neurons[1].position(), neurons[2].position(), neurons[1].type(), 1.1F,
                     1.0F);

    EventQueue queue{};
    TimeManager time{0.5F, 0.0F};
    SimulationEngine engine{neurons, synapses, queue, time};

    engine.inject_event(SpikeEvent{0U, 0U, 0.0F, 1.1F});

    EXPECT_EQ(engine.tick(), 1U);
    EXPECT_NEAR(neurons[0].last_spike_time(), 0.0F, 1e-6F);
    EXPECT_EQ(queue.size(), 1U);

    EXPECT_EQ(engine.tick(), 0U);
    EXPECT_EQ(queue.size(), 1U);

    EXPECT_EQ(engine.tick(), 1U);
    EXPECT_NEAR(neurons[1].last_spike_time(), 1.0F, 1e-6F);
    EXPECT_EQ(queue.size(), 1U);

    EXPECT_EQ(engine.tick(), 0U);
    EXPECT_EQ(queue.size(), 1U);

    EXPECT_EQ(engine.tick(), 1U);
    EXPECT_NEAR(neurons[2].last_spike_time(), 2.0F, 1e-6F);
    EXPECT_EQ(queue.size(), 0U);
}

TEST(SimulationEngineTest, AdvancesTimeOnEmptyTick) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::engine::EventQueue;
    using senna::core::engine::SimulationEngine;
    using senna::core::engine::TimeManager;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory);
    SynapseStore synapses{neurons.size()};

    EventQueue queue{};
    TimeManager time{0.5F, 0.0F};
    SimulationEngine engine{neurons, synapses, queue, time};

    EXPECT_EQ(engine.tick(), 0U);
    EXPECT_NEAR(time.elapsed(), 0.5F, 1e-6F);
}

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "core/domain/types.h"
#include "core/engine/network_builder.h"

namespace {

senna::core::engine::NetworkBuilderConfig make_wave_config() {
    senna::core::engine::NetworkBuilderConfig config{};
    config.lattice.width = 4U;
    config.lattice.height = 4U;
    config.lattice.depth = 3U;
    config.lattice.processing_density = 1.0F;
    config.lattice.excitatory_ratio = 1.0F;
    config.lattice.neighbor_radius = 1.1F;
    config.lattice.output_neurons = 10U;
    config.min_weight = 1.1F;
    config.max_weight = 1.1F;
    config.input_spike_value = 1.1F;
    config.dt = 0.5F;
    config.c_base = 1.0F;
    return config;
}

std::size_t sum_spikes(const std::vector<std::size_t>& trace) {
    return std::accumulate(trace.begin(), trace.end(), std::size_t{0U});
}

std::vector<senna::core::domain::NeuronId> sensor_neuron_ids(
    const senna::core::engine::Network& network) {
    std::vector<senna::core::domain::NeuronId> ids{};
    for (const auto& neuron : network.lattice().neurons()) {
        if (neuron.position().z == 0U) {
            ids.push_back(neuron.id());
        }
    }
    return ids;
}

}  // namespace

TEST(NetworkBuilderTest, BuildsConnectedNetworkForConfiguredLattice) {
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto network = builder.build(42U);

    EXPECT_GT(network.lattice().neuron_count(), 0U);
    EXPECT_GT(network.synapse_count(), 0U);
}

TEST(NetworkBuilderTest, NoStimulusKeepsNetworkSilent) {
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto network = builder.build(7U);

    const auto trace = network.simulate(3.0F);
    ASSERT_FALSE(trace.empty());
    EXPECT_EQ(sum_spikes(trace), 0U);
}

TEST(NetworkBuilderTest, SingleStimulusProducesWaveBeyondFirstTick) {
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto network = builder.build(7U);
    const auto sensor_ids = sensor_neuron_ids(network);

    ASSERT_FALSE(sensor_ids.empty());
    network.inject_spike(sensor_ids.front(), 0.0F);

    const auto trace = network.simulate(3.0F);
    ASSERT_GT(trace.size(), 1U);
    EXPECT_GT(sum_spikes(trace), 0U);
    EXPECT_TRUE(std::any_of(trace.begin() + 1, trace.end(),
                            [](const std::size_t spikes) { return spikes > 0U; }));
}

TEST(NetworkBuilderTest, TenStimuliGenerateMoreSpikesThanSingleStimulus) {
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto single = builder.build(11U);
    auto multiple = builder.build(11U);

    const auto sensor_ids = sensor_neuron_ids(single);
    ASSERT_GE(sensor_ids.size(), 10U);

    single.inject_spike(sensor_ids[0], 0.0F);
    for (std::size_t i = 0U; i < 10U; ++i) {
        multiple.inject_spike(sensor_ids[i], 0.0F);
    }

    const auto single_trace = single.simulate(3.0F);
    const auto multiple_trace = multiple.simulate(3.0F);

    EXPECT_GT(sum_spikes(multiple_trace), sum_spikes(single_trace));
}

TEST(NetworkBuilderTest, ProducesDeterministicTraceForSameSeed) {
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto network_a = builder.build(2026U);
    auto network_b = builder.build(2026U);

    const auto sensor_ids = sensor_neuron_ids(network_a);
    ASSERT_GE(sensor_ids.size(), 5U);

    for (std::size_t i = 0U; i < 5U; ++i) {
        network_a.inject_spike(sensor_ids[i], 0.0F);
        network_b.inject_spike(sensor_ids[i], 0.0F);
    }

    const auto trace_a = network_a.simulate(4.0F);
    const auto trace_b = network_b.simulate(4.0F);

    EXPECT_EQ(network_a.synapse_count(), network_b.synapse_count());
    EXPECT_EQ(trace_a, trace_b);
}

TEST(NetworkBuilderTest, ResetBetweenSamplesClearsTransientNeuronStateOnly) {
    using senna::core::domain::SpikeEvent;
    using senna::core::engine::NetworkBuilder;

    const NetworkBuilder builder{make_wave_config()};
    auto network = builder.build(17U);

    auto& neurons = network.neurons();
    ASSERT_FALSE(neurons.empty());

    neurons.front().set_threshold(1.7F);
    neurons.front().set_average_rate(12.0F);

    network.inject_event(SpikeEvent{
        neurons.front().id(),
        neurons.front().id(),
        0.25F,
        0.5F,
    });
    static_cast<void>(network.tick());

    ASSERT_GT(neurons.front().potential(), 0.0F);
    EXPECT_GT(neurons.front().last_update_time(), 0.0F);

    network.reset_between_samples();

    EXPECT_FLOAT_EQ(neurons.front().potential(), neurons.front().config().v_rest);
    EXPECT_FLOAT_EQ(neurons.front().threshold(), 1.7F);
    EXPECT_FLOAT_EQ(neurons.front().average_rate(), 12.0F);
    EXPECT_FLOAT_EQ(neurons.front().last_update_time(), 0.0F);
    EXPECT_TRUE(std::isinf(neurons.front().last_spike_time()));
    EXPECT_TRUE(std::signbit(neurons.front().last_spike_time()));
    EXPECT_FALSE(neurons.front().in_refractory());
}

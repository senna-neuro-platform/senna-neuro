#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "core/domain/lattice.h"
#include "core/engine/network_builder.h"
#include "core/io/first_spike_decoder.h"
#include "core/io/rate_encoder.h"
#include "test_support/require_value.h"

namespace {

senna::core::domain::Lattice make_sensor_lattice() {
    senna::core::domain::LatticeConfig config{};
    config.width = 28U;
    config.height = 28U;
    config.depth = 2U;
    config.processing_density = 0.0F;
    config.output_neurons = 10U;
    config.neighbor_radius = 1.0F;

    std::mt19937 random{42U};
    senna::core::domain::Lattice lattice{config};
    lattice.generate(random);
    return lattice;
}

std::vector<senna::core::domain::NeuronId> output_ids_from_lattice(
    const senna::core::domain::Lattice& lattice) {
    std::vector<senna::core::domain::NeuronId> output_ids{};
    const auto output_z = static_cast<std::uint16_t>(lattice.config().depth - 1U);
    for (const auto& neuron : lattice.neurons()) {
        if (neuron.position().z == output_z) {
            output_ids.push_back(neuron.id());
        }
    }
    std::sort(output_ids.begin(), output_ids.end());
    return output_ids;
}

senna::core::io::MnistImage image_with_single_pixel(const std::uint8_t value) {
    senna::core::io::MnistImage image{};
    image.fill(0U);
    image[0] = value;
    return image;
}

}  // namespace

TEST(RateEncoderTest, EncodesBlackWhiteAndMediumPixelRates) {
    using senna::core::io::RateEncoder;
    using senna::core::io::RateEncoderConfig;

    const auto lattice = make_sensor_lattice();
    const auto sensor_id = lattice.neuron_id_at(senna::core::domain::Coord3D{0U, 0U, 0U});
    ASSERT_TRUE(sensor_id.has_value());
    const auto sensor = require_value(sensor_id, "sensor neuron must exist");

    RateEncoderConfig config{};
    config.max_rate_hz = 100.0F;
    config.dt = 0.5F;
    config.seed = 42U;

    RateEncoder encoder_black{lattice, config};
    RateEncoder encoder_medium{lattice, config};
    RateEncoder encoder_white{lattice, config};

    const auto black_spikes = encoder_black.encode(image_with_single_pixel(0U), 50.0F);
    const auto medium_spikes = encoder_medium.encode(image_with_single_pixel(128U), 50.0F);
    const auto white_spikes = encoder_white.encode(image_with_single_pixel(255U), 50.0F);

    EXPECT_TRUE(black_spikes.empty());
    EXPECT_GE(medium_spikes.size(), 1U);
    EXPECT_LE(medium_spikes.size(), 8U);
    EXPECT_GE(white_spikes.size(), 3U);
    EXPECT_LE(white_spikes.size(), 12U);
    EXPECT_GE(white_spikes.size(), medium_spikes.size());

    for (const auto& spike : white_spikes) {
        EXPECT_EQ(spike.source, sensor);
        EXPECT_EQ(spike.target, sensor);
        EXPECT_GE(spike.arrival, 0.0F);
        EXPECT_LT(spike.arrival, 50.0F);
    }
}

TEST(FirstSpikeDecoderTest, ReturnsMinusOneWhenNoOutputSpikes) {
    using senna::core::io::FirstSpikeDecoder;

    const std::vector<senna::core::domain::NeuronId> output_ids{
        100U, 101U, 102U, 103U, 104U, 105U, 106U, 107U, 108U, 109U,
    };
    const FirstSpikeDecoder decoder{output_ids};
    EXPECT_EQ(decoder.decode({}), -1);
}

TEST(FirstSpikeDecoderTest, DecodesEarliestSpikeAndGeneratesWtaEvents) {
    using senna::core::domain::SpikeEvent;
    using senna::core::io::FirstSpikeDecoder;

    const std::vector<senna::core::domain::NeuronId> output_ids{
        100U, 101U, 102U, 103U, 104U, 105U, 106U, 107U, 108U, 109U,
    };
    const FirstSpikeDecoder decoder{output_ids, 7.0F};

    const std::vector<SpikeEvent> spikes{
        {103U, 103U, 2.0F, 1.0F},
        {101U, 101U, 1.0F, 1.0F},
        {107U, 107U, 3.0F, 1.0F},
    };

    EXPECT_EQ(decoder.decode(spikes), 1);

    const auto wta = decoder.winner_take_all_events(101U, 1.0F);
    ASSERT_EQ(wta.size(), 9U);
    for (const auto& event : wta) {
        EXPECT_EQ(event.source, 101U);
        EXPECT_NE(event.target, 101U);
        EXPECT_NEAR(event.arrival, 1.0F, 1e-6F);
        EXPECT_NEAR(event.value, -7.0F, 1e-6F);
    }
}

TEST(FirstSpikeDecoderTest, ResolvesTieDeterministicallyByOutputIndex) {
    using senna::core::domain::SpikeEvent;
    using senna::core::io::FirstSpikeDecoder;

    const std::vector<senna::core::domain::NeuronId> output_ids{
        200U, 201U, 202U, 203U, 204U, 205U, 206U, 207U, 208U, 209U,
    };
    const FirstSpikeDecoder decoder{output_ids};

    const std::vector<SpikeEvent> spikes{
        {207U, 207U, 1.0F, 1.0F},
        {202U, 202U, 1.0F, 1.0F},
    };

    EXPECT_EQ(decoder.decode(spikes), 2);
}

TEST(IOPipelineTest, EncodesInputRunsNetworkAndDecodesPrediction) {
    using senna::core::engine::NetworkBuilder;
    using senna::core::engine::NetworkBuilderConfig;
    using senna::core::io::FirstSpikeDecoder;
    using senna::core::io::RateEncoder;
    using senna::core::io::RateEncoderConfig;

    NetworkBuilderConfig network_config{};
    network_config.lattice.width = 28U;
    network_config.lattice.height = 28U;
    network_config.lattice.depth = 3U;
    network_config.lattice.processing_density = 0.35F;
    network_config.lattice.excitatory_ratio = 1.0F;
    network_config.lattice.neighbor_radius = 1.5F;
    network_config.lattice.output_neurons = 10U;
    network_config.dt = 0.5F;
    network_config.c_base = 1.0F;
    network_config.min_weight = 0.2F;
    network_config.max_weight = 0.2F;
    network_config.input_spike_value = 1.1F;

    const NetworkBuilder builder{network_config};
    auto network = builder.build(2026U);

    RateEncoderConfig encoder_config{};
    encoder_config.max_rate_hz = 120.0F;
    encoder_config.dt = 0.5F;
    encoder_config.seed = 2026U;
    RateEncoder encoder{network.lattice(), encoder_config};

    auto image = image_with_single_pixel(255U);
    const auto input_events = encoder.encode(image, 10.0F);
    for (const auto& event : input_events) {
        network.inject_spike(event.target, event.arrival);
    }

    const auto output_ids = output_ids_from_lattice(network.lattice());
    ASSERT_EQ(output_ids.size(), 10U);
    const FirstSpikeDecoder decoder{output_ids};

    std::vector<senna::core::domain::SpikeEvent> output_spikes{};
    while (network.elapsed() < 10.0F) {
        static_cast<void>(network.tick());
        for (const auto& spike : network.emitted_spikes_last_tick()) {
            if (std::find(output_ids.begin(), output_ids.end(), spike.source) != output_ids.end()) {
                output_spikes.push_back(spike);
            }
        }
    }

    const int prediction = decoder.decode(output_spikes);
    EXPECT_GE(prediction, -1);
    EXPECT_LT(prediction, 10);
}

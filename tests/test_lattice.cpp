#include <gtest/gtest.h>

#include <optional>
#include <random>
#include <stdexcept>

#include "core/domain/lattice.h"

namespace {

template <typename T>
T require_value(const std::optional<T>& value, const char* message) {
    if (!value.has_value()) {
        throw std::runtime_error(message);
    }
    return *value;
}

}  // namespace

TEST(LatticeTest, BuildsExpectedMvpGridAndLayerCounts) {
    using senna::core::domain::Lattice;

    std::mt19937 random{42U};
    Lattice lattice{};
    lattice.generate(random);

    constexpr std::size_t kWidth = 28U;
    constexpr std::size_t kHeight = 28U;
    constexpr std::size_t kDepth = 20U;
    constexpr std::size_t kSensorCount = kWidth * kHeight;
    constexpr std::size_t kProcessingDepth = 18U;
    constexpr std::size_t kProcessingVoxelCount = kWidth * kHeight * kProcessingDepth;

    EXPECT_EQ(lattice.voxel_count(), kWidth * kHeight * kDepth);
    EXPECT_EQ(lattice.sensor_neuron_count(), kSensorCount);
    EXPECT_EQ(lattice.output_neuron_count(), 10U);
    EXPECT_EQ(lattice.neuron_count(), lattice.sensor_neuron_count() +
                                          lattice.processing_neuron_count() +
                                          lattice.output_neuron_count());

    const auto processing_density = static_cast<double>(lattice.processing_neuron_count()) /
                                    static_cast<double>(kProcessingVoxelCount);
    EXPECT_GE(processing_density, 0.65);
    EXPECT_LE(processing_density, 0.75);
}

TEST(LatticeTest, SensorAndOutputLayersAreExcitatory) {
    using senna::core::domain::Lattice;
    using senna::core::domain::NeuronType;

    std::mt19937 random{1337U};
    Lattice lattice{};
    lattice.generate(random);

    std::size_t sensor_count = 0U;
    std::size_t output_count = 0U;
    const auto output_z = static_cast<std::uint16_t>(lattice.config().depth - 1U);

    for (const auto& neuron : lattice.neurons()) {
        if (neuron.position().z == 0U) {
            ++sensor_count;
            EXPECT_EQ(neuron.type(), NeuronType::Excitatory);
        }
        if (neuron.position().z == output_z) {
            ++output_count;
            EXPECT_EQ(neuron.type(), NeuronType::Excitatory);
        }
    }

    EXPECT_EQ(sensor_count, lattice.sensor_neuron_count());
    EXPECT_EQ(output_count, lattice.output_neuron_count());
}

TEST(LatticeTest, FindsExpectedNeighborCountsInCenterAndCorner) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Lattice;
    using senna::core::domain::LatticeConfig;

    LatticeConfig config{};
    config.processing_density = 1.0F;
    config.neighbor_radius = 2.0F;

    std::mt19937 random{7U};
    Lattice lattice{config};
    lattice.generate(random);

    const auto center_id = lattice.neuron_id_at(Coord3D{14U, 14U, 10U});
    ASSERT_TRUE(center_id.has_value());
    const auto center = require_value(center_id, "center neuron must exist");

    const auto center_neighbors = lattice.neighbors(center);
    EXPECT_EQ(center_neighbors.size(), 32U);
    for (const auto& neighbor : center_neighbors) {
        EXPECT_LE(neighbor.distance, 2.0F);
    }

    const auto corner_id = lattice.neuron_id_at(Coord3D{0U, 0U, 0U});
    ASSERT_TRUE(corner_id.has_value());
    const auto corner = require_value(corner_id, "corner neuron must exist");

    const auto corner_neighbors = lattice.neighbors(corner, 2.0F);
    EXPECT_EQ(corner_neighbors.size(), 10U);
    EXPECT_LT(corner_neighbors.size(), center_neighbors.size());
}

TEST(LatticeTest, IsDeterministicForSameSeed) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Lattice;

    std::mt19937 random_a{2026U};
    std::mt19937 random_b{2026U};

    Lattice lattice_a{};
    Lattice lattice_b{};
    lattice_a.generate(random_a);
    lattice_b.generate(random_b);

    ASSERT_EQ(lattice_a.voxel_count(), lattice_b.voxel_count());
    ASSERT_EQ(lattice_a.neuron_count(), lattice_b.neuron_count());

    const auto width = lattice_a.config().width;
    const auto height = lattice_a.config().height;
    const auto depth = lattice_a.config().depth;

    for (std::uint16_t z = 0U; z < depth; ++z) {
        for (std::uint16_t y = 0U; y < height; ++y) {
            for (std::uint16_t x = 0U; x < width; ++x) {
                const Coord3D position{x, y, z};
                const auto id_a = lattice_a.neuron_id_at(position);
                const auto id_b = lattice_b.neuron_id_at(position);

                ASSERT_EQ(id_a.has_value(), id_b.has_value());
                if (!id_a.has_value()) {
                    continue;
                }

                const auto neuron_id_a = require_value(id_a, "lattice_a neuron id must exist");
                const auto neuron_id_b = require_value(id_b, "lattice_b neuron id must exist");
                EXPECT_EQ(neuron_id_a, neuron_id_b);
                EXPECT_EQ(lattice_a.neurons().at(neuron_id_a).type(),
                          lattice_b.neurons().at(neuron_id_b).type());
            }
        }
    }
}

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "core/domain/types.h"

namespace {

bool almost_equal(const float lhs, const float rhs, const float eps = 1e-5F) {
    return std::fabs(lhs - rhs) <= eps;
}

}  // namespace

TEST(Coord3DistanceTest, Calculates2DDistance) {
    using senna::core::domain::Coord3D;

    const Coord3D origin{0, 0, 0};
    const Coord3D point{3, 4, 0};

    EXPECT_TRUE(almost_equal(origin.distance(point), 5.0F));
}

TEST(Coord3DistanceTest, Calculates3DDistance) {
    using senna::core::domain::Coord3D;

    const Coord3D origin{0, 0, 0};
    const Coord3D point_3d{2, 3, 6};

    EXPECT_TRUE(almost_equal(origin.distance(point_3d), 7.0F));
}

TEST(Coord3DistanceTest, SupportsMixedCoordinateTypes) {
    using senna::core::domain::Coord3;
    using senna::core::domain::Coord3D;

    const Coord3D origin{0, 0, 0};
    const Coord3<int> int_point{3, 4, 0};

    EXPECT_TRUE(almost_equal(origin.distance(int_point), 5.0F));
}

TEST(SpikeEventTest, OrdersByArrival) {
    using senna::core::domain::SpikeEvent;

    const SpikeEvent early{1U, 2U, 1.0F, 0.25F};
    const SpikeEvent late{1U, 2U, 2.0F, 0.25F};

    EXPECT_LT(early, late);
    EXPECT_FALSE(late < early);
    EXPECT_GT(late, early);
}

TEST(SpikeEventTest, TemplateComparatorOrdersByArrival) {
    using senna::core::domain::ArrivalEarlier;
    using senna::core::domain::SpikeEvent;

    const SpikeEvent early{1U, 2U, 1.0F, 0.25F};
    const SpikeEvent late{1U, 2U, 2.0F, 0.25F};
    const ArrivalEarlier<SpikeEvent> compare_by_arrival{};

    EXPECT_TRUE(compare_by_arrival(early, late));
    EXPECT_FALSE(compare_by_arrival(late, early));
}

TEST(NeuronTypeTest, DistinguishesExcitatoryAndInhibitory) {
    using senna::core::domain::NeuronType;

    EXPECT_NE(NeuronType::Excitatory, NeuronType::Inhibitory);
    EXPECT_NE(static_cast<std::uint8_t>(NeuronType::Excitatory),
              static_cast<std::uint8_t>(NeuronType::Inhibitory));
}

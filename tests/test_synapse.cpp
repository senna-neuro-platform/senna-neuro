#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "core/domain/synapse.h"

namespace {

senna::core::domain::Coord3D coord_from_id(const senna::core::domain::NeuronId id) {
    constexpr std::uint32_t kWidth = 28U;
    constexpr std::uint32_t kHeight = 28U;
    constexpr std::uint32_t kDepth = 20U;
    constexpr std::uint32_t kLayerSize = kWidth * kHeight;

    const auto x = static_cast<std::uint16_t>(id % kWidth);
    const auto y = static_cast<std::uint16_t>((id / kWidth) % kHeight);
    const auto z = static_cast<std::uint16_t>((id / kLayerSize) % kDepth);
    return {x, y, z};
}

}  // namespace

TEST(SynapseTest, DelayIsProportionalToDistance) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    SynapseStore store{};
    const auto id = store.connect(1U, 2U, Coord3D{0, 0, 0}, Coord3D{0, 0, 2},
                                  NeuronType::Excitatory, 0.05F, 1.0F);
    const auto& synapse = store.at(id);

    EXPECT_NEAR(synapse.delay, 2.0F, 1e-5F);
}

TEST(SynapseTest, SignDependsOnPresynapticNeuronType) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    SynapseStore store{};
    const auto exc_id =
        store.connect(1U, 3U, Coord3D{0, 0, 0}, Coord3D{1, 0, 0}, NeuronType::Excitatory, 0.1F);
    const auto inh_id =
        store.connect(2U, 3U, Coord3D{0, 0, 0}, Coord3D{1, 0, 0}, NeuronType::Inhibitory, 0.1F);

    EXPECT_EQ(store.at(exc_id).sign, 1);
    EXPECT_EQ(store.at(inh_id).sign, -1);
    EXPECT_GT(store.at(exc_id).effective_weight(), 0.0F);
    EXPECT_LT(store.at(inh_id).effective_weight(), 0.0F);
}

TEST(SynapseTest, RandomWeightIsInExpectedRange) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    std::mt19937 random{42U};
    SynapseStore store{};

    const auto id = store.connect_random(1U, 2U, Coord3D{0, 0, 0}, Coord3D{1, 1, 1},
                                         NeuronType::Excitatory, random, 1.0F, 0.01F, 0.1F);

    const auto& synapse = store.at(id);
    EXPECT_GE(synapse.weight, 0.01F);
    EXPECT_LE(synapse.weight, 0.1F);
}

TEST(SynapseStoreTest, BuildsOutgoingAndIncomingIndexes) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    SynapseStore store{4U};
    const auto id0 =
        store.connect(0U, 1U, Coord3D{0, 0, 0}, Coord3D{1, 0, 0}, NeuronType::Excitatory, 0.05F);
    const auto id1 =
        store.connect(0U, 2U, Coord3D{0, 0, 0}, Coord3D{0, 1, 0}, NeuronType::Excitatory, 0.05F);
    const auto id2 =
        store.connect(3U, 2U, Coord3D{0, 0, 0}, Coord3D{0, 0, 1}, NeuronType::Inhibitory, 0.07F);

    EXPECT_EQ(store.outgoing(0U), std::vector<senna::core::domain::SynapseId>({id0, id1}));
    EXPECT_EQ(store.incoming(2U), std::vector<senna::core::domain::SynapseId>({id1, id2}));
    EXPECT_EQ(store.outgoing(3U), std::vector<senna::core::domain::SynapseId>({id2}));
    EXPECT_EQ(store.incoming(1U), std::vector<senna::core::domain::SynapseId>({id0}));

    const auto outgoing_span = store.outgoing_span(0U);
    const auto incoming_span = store.incoming_span(2U);
    EXPECT_EQ(
        std::vector<senna::core::domain::SynapseId>(outgoing_span.begin(), outgoing_span.end()),
        std::vector<senna::core::domain::SynapseId>({id0, id1}));
    EXPECT_EQ(
        std::vector<senna::core::domain::SynapseId>(incoming_span.begin(), incoming_span.end()),
        std::vector<senna::core::domain::SynapseId>({id1, id2}));
}

TEST(SynapseStoreTest, SupportsMvpScaleIndexing) {
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    constexpr std::uint32_t kNeuronCount = 10'000U;
    constexpr std::uint32_t kOutgoingPerNeuron = 30U;

    SynapseStore store{kNeuronCount};
    for (std::uint32_t pre = 0; pre < kNeuronCount; ++pre) {
        const auto pre_pos = coord_from_id(pre);
        for (std::uint32_t offset = 1; offset <= kOutgoingPerNeuron; ++offset) {
            const auto post = static_cast<std::uint32_t>((pre + offset) % kNeuronCount);
            const auto post_pos = coord_from_id(post);
            store.connect(pre, post, pre_pos, post_pos, NeuronType::Excitatory, 0.05F);
        }
    }

    EXPECT_EQ(store.size(), static_cast<std::size_t>(kNeuronCount) * kOutgoingPerNeuron);
    EXPECT_EQ(store.outgoing(0U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.outgoing(9999U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.incoming(0U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.incoming(9999U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.outgoing_span(0U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.outgoing_span(9999U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.incoming_span(0U).size(), kOutgoingPerNeuron);
    EXPECT_EQ(store.incoming_span(9999U).size(), kOutgoingPerNeuron);
}

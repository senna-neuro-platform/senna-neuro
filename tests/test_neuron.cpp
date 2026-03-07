#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

#include "core/domain/neuron.h"

namespace {

bool almost_equal(const float lhs, const float rhs, const float eps = 1e-4F) {
    return std::fabs(lhs - rhs) <= eps;
}

void expect_time_equal(const float lhs, const float rhs, const float eps = 1e-4F) {
    if (std::isinf(lhs) || std::isinf(rhs)) {
        EXPECT_TRUE(std::isinf(lhs));
        EXPECT_TRUE(std::isinf(rhs));
        EXPECT_EQ(std::signbit(lhs), std::signbit(rhs));
        return;
    }
    EXPECT_NEAR(lhs, rhs, eps);
}

}  // namespace

TEST(NeuronTest, DecaysPotentialOverMembraneTau) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;

    NeuronConfig config{};
    config.theta_base = 10.0F;
    config.tau_m = 20.0F;
    Neuron neuron{1U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, config};

    EXPECT_FALSE(neuron.receive_input(0.0F, 1.0F).has_value());
    EXPECT_FALSE(neuron.receive_input(config.tau_m, 0.0F).has_value());
    EXPECT_NEAR(neuron.potential(), std::exp(-1.0F), 1e-3F);
}

TEST(NeuronTest, SpikesAndResetsPotential) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronType;

    Neuron neuron{2U, Coord3D{1U, 1U, 1U}, NeuronType::Excitatory};

    EXPECT_FALSE(neuron.receive_input(0.0F, 0.9F).has_value());
    const auto spike = neuron.receive_input(0.0F, 0.2F);

    ASSERT_TRUE(spike.has_value());
    EXPECT_NEAR(spike->arrival, 0.0F, 1e-5F);
    EXPECT_NEAR(spike->value, 1.0F, 1e-5F);
    EXPECT_NEAR(neuron.potential(), 0.0F, 1e-5F);
    EXPECT_NEAR(neuron.last_spike_time(), 0.0F, 1e-5F);
}

TEST(NeuronTest, HonorsRefractoryPeriod) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronType;

    Neuron neuron{3U, Coord3D{2U, 2U, 2U}, NeuronType::Excitatory};

    ASSERT_TRUE(neuron.receive_input(0.0F, 1.1F).has_value());
    EXPECT_FALSE(neuron.receive_input(1.0F, 10.0F).has_value());
    EXPECT_TRUE(neuron.in_refractory());

    const auto after_ref = neuron.receive_input(2.01F, 1.1F);
    ASSERT_TRUE(after_ref.has_value());
    EXPECT_NEAR(after_ref->arrival, 2.01F, 1e-5F);
}

TEST(NeuronTest, EmitsSignedSpikeByNeuronType) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronType;

    Neuron excitatory{4U, Coord3D{3U, 3U, 3U}, NeuronType::Excitatory};
    Neuron inhibitory{5U, Coord3D{4U, 4U, 4U}, NeuronType::Inhibitory};

    const auto e_spike = excitatory.receive_input(0.0F, 1.1F);
    const auto i_spike = inhibitory.receive_input(0.0F, 1.1F);

    ASSERT_TRUE(e_spike.has_value());
    ASSERT_TRUE(i_spike.has_value());
    EXPECT_GT(e_spike->value, 0.0F);
    EXPECT_LT(i_spike->value, 0.0F);
}

TEST(NeuronTest, IsDeterministicForSameInputSequence) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronType;

    const std::vector<std::pair<float, float>> input_sequence{
        {0.0F, 0.4F},  {0.5F, 0.35F}, {1.0F, 0.40F}, {1.5F, 0.10F},
        {2.0F, 1.20F}, {4.5F, 0.95F}, {4.5F, 0.10F},
    };

    Neuron neuron_a{10U, Coord3D{5U, 5U, 5U}, NeuronType::Excitatory};
    Neuron neuron_b{10U, Coord3D{5U, 5U, 5U}, NeuronType::Excitatory};

    for (const auto& [time, value] : input_sequence) {
        const auto spike_a = neuron_a.receive_input(time, value);
        const auto spike_b = neuron_b.receive_input(time, value);

        ASSERT_EQ(spike_a.has_value(), spike_b.has_value());
        if (spike_a.has_value()) {
            EXPECT_NEAR(spike_a->arrival, spike_b->arrival, 1e-5F);
            EXPECT_NEAR(spike_a->value, spike_b->value, 1e-5F);
        }

        EXPECT_TRUE(almost_equal(neuron_a.potential(), neuron_b.potential()));
        expect_time_equal(neuron_a.last_update_time(), neuron_b.last_update_time());
        expect_time_equal(neuron_a.last_spike_time(), neuron_b.last_spike_time());
        EXPECT_TRUE(almost_equal(neuron_a.average_rate(), neuron_b.average_rate()));
        EXPECT_EQ(neuron_a.in_refractory(), neuron_b.in_refractory());
    }
}

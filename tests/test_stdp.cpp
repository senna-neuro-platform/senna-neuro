#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <vector>

#include "core/domain/synapse.h"
#include "core/domain/types.h"
#include "core/plasticity/stdp.h"
#include "core/plasticity/supervisor.h"

namespace {

template <typename T>
T require_value(const std::optional<T>& value, const char* message) {
    if (!value.has_value()) {
        throw std::runtime_error(message);
    }
    return *value;
}

}  // namespace

TEST(STDPRuleTest, CausalPairIncreasesWeight) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::STDPRule;

    SynapseStore synapses{2U};
    const auto synapse_id = synapses.connect(0U, 1U, Coord3D{0U, 0U, 0U}, Coord3D{1U, 0U, 0U},
                                             NeuronType::Excitatory, 0.5F, 1.0F);

    STDPRule stdp{};
    const auto initial = synapses.at(synapse_id).weight;
    stdp.on_pre_spike(0U, 10.0F, synapses);
    stdp.on_post_spike(1U, 15.0F, synapses);

    EXPECT_GT(synapses.at(synapse_id).weight, initial);
}

TEST(STDPRuleTest, AntiCausalPairDecreasesWeight) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::STDPRule;

    SynapseStore synapses{2U};
    const auto synapse_id = synapses.connect(0U, 1U, Coord3D{0U, 0U, 0U}, Coord3D{1U, 0U, 0U},
                                             NeuronType::Excitatory, 0.5F, 1.0F);

    STDPRule stdp{};
    const auto initial = synapses.at(synapse_id).weight;
    stdp.on_post_spike(1U, 10.0F, synapses);
    stdp.on_pre_spike(0U, 15.0F, synapses);

    EXPECT_LT(synapses.at(synapse_id).weight, initial);
}

TEST(STDPRuleTest, LargeDeltaTProducesNearZeroChange) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::STDPRule;

    SynapseStore synapses{2U};
    const auto synapse_id = synapses.connect(0U, 1U, Coord3D{0U, 0U, 0U}, Coord3D{1U, 0U, 0U},
                                             NeuronType::Excitatory, 0.5F, 1.0F);

    STDPRule stdp{};
    const auto initial = synapses.at(synapse_id).weight;
    stdp.on_pre_spike(0U, 0.0F, synapses);
    stdp.on_post_spike(1U, 200.0F, synapses);

    EXPECT_NEAR(synapses.at(synapse_id).weight, initial, 1e-4F);
}

TEST(STDPRuleTest, WeightIsClampedByWMax) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::STDPConfig;
    using senna::core::plasticity::STDPRule;

    SynapseStore synapses{2U};
    const auto synapse_id = synapses.connect(0U, 1U, Coord3D{0U, 0U, 0U}, Coord3D{1U, 0U, 0U},
                                             NeuronType::Excitatory, 0.49F, 1.0F);

    STDPConfig config{};
    config.w_max = 0.5F;
    STDPRule stdp{config};

    for (int i = 0; i < 100; ++i) {
        const auto t = static_cast<float>(i) * 10.0F;
        stdp.on_pre_spike(0U, t, synapses);
        stdp.on_post_spike(1U, t + 1.0F, synapses);
    }

    EXPECT_LE(synapses.at(synapse_id).weight, 0.5F);
    EXPECT_GE(synapses.at(synapse_id).weight, 0.0F);
}

TEST(SupervisorTest, ReinforcementGrowsWeightToCorrectOutput) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::STDPRule;
    using senna::core::plasticity::Supervisor;

    SynapseStore synapses{3U};
    const auto to_correct = synapses.connect(0U, 1U, Coord3D{0U, 0U, 0U}, Coord3D{1U, 0U, 0U},
                                             NeuronType::Excitatory, 0.1F, 1.0F);
    const auto to_wrong = synapses.connect(0U, 2U, Coord3D{0U, 0U, 0U}, Coord3D{2U, 0U, 0U},
                                           NeuronType::Excitatory, 0.1F, 1.0F);

    STDPRule stdp{};
    Supervisor supervisor{};
    const std::vector<senna::core::domain::NeuronId> output_neurons{1U, 2U};

    for (int sample = 0; sample < 10; ++sample) {
        const auto t_pre = static_cast<float>(sample) * 50.0F;
        stdp.on_pre_spike(0U, t_pre, synapses);

        const auto correction = supervisor.correction_event(
            /*predicted_class=*/1, /*expected_class=*/0, output_neurons, t_pre + 1.0F);
        ASSERT_TRUE(correction.has_value());
        const auto correction_event = require_value(correction, "expected supervision event");
        stdp.on_post_spike(correction_event.target, correction_event.arrival, synapses);
    }

    EXPECT_GT(synapses.at(to_correct).weight, synapses.at(to_wrong).weight);
    EXPECT_GT(synapses.at(to_correct).weight, 0.1F);
}

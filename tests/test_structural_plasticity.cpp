#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "core/domain/lattice.h"
#include "core/domain/synapse.h"
#include "core/plasticity/structural_plasticity.h"

namespace {

senna::core::domain::Lattice make_test_lattice() {
    senna::core::domain::LatticeConfig config{};
    config.width = 4U;
    config.height = 4U;
    config.depth = 3U;
    config.processing_density = 1.0F;
    config.excitatory_ratio = 1.0F;
    config.output_neurons = 4U;
    config.neighbor_radius = 1.6F;

    std::mt19937 random{42U};
    senna::core::domain::Lattice lattice{config};
    lattice.generate(random);
    return lattice;
}

senna::core::domain::NeuronId find_neuron_with_neighbors(
    const senna::core::domain::Lattice& lattice, const float radius) {
    for (const auto& neuron : lattice.neurons()) {
        if (!lattice.neighbors(neuron.id(), radius).empty()) {
            return neuron.id();
        }
    }
    throw std::runtime_error("No neuron with neighbors found");
}

void expect_index_consistency(const senna::core::domain::SynapseStore& synapses,
                              const std::size_t neuron_count) {
    for (std::size_t neuron = 0U; neuron < neuron_count; ++neuron) {
        for (const auto synapse_id :
             synapses.outgoing(static_cast<senna::core::domain::NeuronId>(neuron))) {
            ASSERT_LT(static_cast<std::size_t>(synapse_id), synapses.size());
            EXPECT_EQ(synapses.at(synapse_id).pre_id,
                      static_cast<senna::core::domain::NeuronId>(neuron));
        }

        for (const auto synapse_id :
             synapses.incoming(static_cast<senna::core::domain::NeuronId>(neuron))) {
            ASSERT_LT(static_cast<std::size_t>(synapse_id), synapses.size());
            EXPECT_EQ(synapses.at(synapse_id).post_id,
                      static_cast<senna::core::domain::NeuronId>(neuron));
        }
    }
}

}  // namespace

TEST(StructuralPlasticityTest, PrunesWeakAndKeepsStrongSynapses) {
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::StructuralPlasticity;
    using senna::core::plasticity::StructuralPlasticityConfig;

    auto lattice = make_test_lattice();
    auto& neurons = lattice.neurons();

    SynapseStore synapses{lattice.neuron_count()};
    const auto weak_pre = neurons[0U];
    const auto weak_post = neurons[1U];
    const auto strong_pre = neurons[2U];
    const auto strong_post = neurons[3U];

    synapses.connect(weak_pre.id(), weak_post.id(), weak_pre.position(), weak_post.position(),
                     weak_pre.type(), 0.0005F, 1.0F);
    synapses.connect(strong_pre.id(), strong_post.id(), strong_pre.position(),
                     strong_post.position(), strong_pre.type(), 0.5F, 1.0F);

    for (auto& neuron : neurons) {
        neuron.set_average_rate(10.0F);
    }

    StructuralPlasticityConfig config{};
    config.w_min = 0.001F;
    config.r_target_hz = 0.0F;
    config.update_interval_ticks = 1U;

    StructuralPlasticity rule{config};
    const auto stats = rule.run_once(lattice, neurons, synapses);

    EXPECT_EQ(stats.pruned, 1U);
    EXPECT_EQ(stats.sprouted, 0U);
    ASSERT_EQ(synapses.size(), 1U);
    EXPECT_EQ(synapses.at(0U).pre_id, strong_pre.id());
    EXPECT_EQ(synapses.at(0U).post_id, strong_post.id());
    EXPECT_NEAR(synapses.at(0U).weight, 0.5F, 1e-6F);
}

TEST(StructuralPlasticityTest, QuietNeuronGetsNewIncomingSynapses) {
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::StructuralPlasticity;
    using senna::core::plasticity::StructuralPlasticityConfig;

    auto lattice = make_test_lattice();
    auto& neurons = lattice.neurons();

    for (auto& neuron : neurons) {
        neuron.set_average_rate(10.0F);
    }

    constexpr float kSproutRadius = 1.6F;
    const auto quiet_id = find_neuron_with_neighbors(lattice, kSproutRadius);
    neurons[static_cast<std::size_t>(quiet_id)].set_average_rate(0.0F);

    SynapseStore synapses{lattice.neuron_count()};

    StructuralPlasticityConfig config{};
    config.w_min = 0.0F;
    config.r_target_hz = 5.0F;
    config.quiet_ratio = 0.5F;
    config.sprout_radius = kSproutRadius;
    config.sprout_weight = 0.01F;
    config.max_sprouts_per_neuron = 2U;
    config.update_interval_ticks = 1U;

    StructuralPlasticity rule{config};
    const auto stats = rule.run_once(lattice, neurons, synapses);

    EXPECT_EQ(stats.pruned, 0U);
    EXPECT_GT(stats.sprouted, 0U);

    const auto& incoming = synapses.incoming(quiet_id);
    ASSERT_FALSE(incoming.empty());
    EXPECT_LE(incoming.size(), config.max_sprouts_per_neuron);
    for (const auto synapse_id : incoming) {
        EXPECT_NEAR(synapses.at(synapse_id).weight, config.sprout_weight, 1e-6F);
    }
}

TEST(StructuralPlasticityTest, PruneAndSproutKeepSynapseCountApproximatelyStable) {
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::StructuralPlasticity;
    using senna::core::plasticity::StructuralPlasticityConfig;

    auto lattice = make_test_lattice();
    auto& neurons = lattice.neurons();

    for (auto& neuron : neurons) {
        neuron.set_average_rate(10.0F);
    }

    constexpr float kSproutRadius = 1.6F;
    const auto quiet_id = find_neuron_with_neighbors(lattice, kSproutRadius);
    const auto neighbors = lattice.neighbors(quiet_id, kSproutRadius);
    ASSERT_FALSE(neighbors.empty());

    const auto weak_pre_id = neighbors.front().id;
    neurons[static_cast<std::size_t>(quiet_id)].set_average_rate(0.0F);

    SynapseStore synapses{lattice.neuron_count()};
    const auto& weak_pre = neurons[static_cast<std::size_t>(weak_pre_id)];
    const auto& quiet_neuron = neurons[static_cast<std::size_t>(quiet_id)];
    synapses.connect(weak_pre.id(), quiet_neuron.id(), weak_pre.position(), quiet_neuron.position(),
                     weak_pre.type(), 0.0005F, 1.0F);

    const auto& strong_pre = neurons[0U];
    const auto& strong_post = neurons[1U];
    synapses.connect(strong_pre.id(), strong_post.id(), strong_pre.position(),
                     strong_post.position(), strong_pre.type(), 0.5F, 1.0F);

    const auto before = synapses.size();

    StructuralPlasticityConfig config{};
    config.w_min = 0.001F;
    config.r_target_hz = 5.0F;
    config.quiet_ratio = 0.5F;
    config.sprout_radius = kSproutRadius;
    config.sprout_weight = 0.01F;
    config.max_sprouts_per_neuron = 1U;
    config.update_interval_ticks = 1U;

    StructuralPlasticity rule{config};
    const auto stats = rule.run_once(lattice, neurons, synapses);

    EXPECT_EQ(stats.pruned, 1U);
    EXPECT_EQ(stats.sprouted, 1U);
    EXPECT_EQ(synapses.size(), before);
    expect_index_consistency(synapses, lattice.neuron_count());
}

TEST(StructuralPlasticityTest, RunsOnlyEveryConfiguredTickInterval) {
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::StructuralPlasticity;
    using senna::core::plasticity::StructuralPlasticityConfig;

    auto lattice = make_test_lattice();
    auto& neurons = lattice.neurons();

    for (auto& neuron : neurons) {
        neuron.set_average_rate(10.0F);
    }

    SynapseStore synapses{lattice.neuron_count()};
    const auto& pre = neurons[0U];
    const auto& post = neurons[1U];
    synapses.connect(pre.id(), post.id(), pre.position(), post.position(), pre.type(), 0.0005F,
                     1.0F);

    StructuralPlasticityConfig config{};
    config.w_min = 0.001F;
    config.r_target_hz = 0.0F;
    config.update_interval_ticks = 3U;

    StructuralPlasticity rule{config};

    const auto first = rule.on_tick(lattice, neurons, synapses);
    const auto second = rule.on_tick(lattice, neurons, synapses);
    EXPECT_EQ(first.pruned, 0U);
    EXPECT_EQ(second.pruned, 0U);
    EXPECT_EQ(synapses.size(), 1U);

    const auto third = rule.on_tick(lattice, neurons, synapses);
    EXPECT_EQ(third.pruned, 1U);
    EXPECT_EQ(synapses.size(), 0U);
}

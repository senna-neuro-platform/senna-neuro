#include <gtest/gtest.h>

#include <vector>

#include "core/domain/synapse.h"
#include "core/plasticity/supervisor.h"
#include "test_support/require_value.h"

TEST(SupervisorTest, EmitsCorrectionSpikeOnlyForIncorrectPrediction) {
    using senna::core::domain::NeuronId;
    using senna::core::plasticity::Supervisor;

    Supervisor supervisor{{1.25F}};
    const std::vector<NeuronId> output_neurons{10U, 11U, 12U};

    const auto no_correction = supervisor.correction_event(1, 1, output_neurons, 5.0F);
    EXPECT_FALSE(no_correction.has_value());

    const auto correction = supervisor.correction_event(0, 2, output_neurons, 7.5F);
    ASSERT_TRUE(correction.has_value());
    const auto correction_event =
        require_value(correction, "expected correction event for incorrect prediction");
    EXPECT_EQ(correction_event.source, 12U);
    EXPECT_EQ(correction_event.target, 12U);
    EXPECT_FLOAT_EQ(correction_event.arrival, 7.5F);
    EXPECT_FLOAT_EQ(correction_event.value, 1.25F);
}

TEST(SupervisorTest, UpdatesExistingIncomingSynapsesUsingObservedActivity) {
    using senna::core::domain::NeuronId;
    using senna::core::domain::Synapse;
    using senna::core::domain::SynapseStore;
    using senna::core::plasticity::Supervisor;

    Supervisor supervisor{};
    SynapseStore synapses{4U};

    const auto expected_exc = synapses.add(Synapse{0U, 2U, 0.50F, 1.0F, 1});
    const auto expected_inh = synapses.add(Synapse{1U, 2U, 0.50F, 1.0F, -1});
    const auto predicted_exc = synapses.add(Synapse{0U, 3U, 0.50F, 1.0F, 1});
    const auto predicted_inh = synapses.add(Synapse{1U, 3U, 0.50F, 1.0F, -1});

    const std::vector<NeuronId> output_neurons{2U, 3U};
    const std::vector<std::uint16_t> pre_spike_counts{4U, 4U, 0U, 0U};

    const auto updated = supervisor.apply_output_weight_update(
        1, 0, output_neurons, pre_spike_counts, synapses, 0.04F, 0.01F, 1.0F);

    EXPECT_EQ(updated, 4U);
    EXPECT_FLOAT_EQ(synapses.at(expected_exc).weight, 0.52F);
    EXPECT_FLOAT_EQ(synapses.at(expected_inh).weight, 0.48F);
    EXPECT_FLOAT_EQ(synapses.at(predicted_exc).weight, 0.48F);
    EXPECT_FLOAT_EQ(synapses.at(predicted_inh).weight, 0.52F);
}

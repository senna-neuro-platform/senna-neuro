#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/domain/types.h"
#include "core/engine/event_queue.h"
#include "core/engine/simulation_engine.h"
#include "core/engine/time_manager.h"
#include "core/persistence/epoch_artifact_pipeline.h"
#include "core/persistence/hdf5_writer.h"
#include "core/persistence/state_serializer.h"

namespace {

std::filesystem::path make_temp_h5_path(const std::string& prefix) {
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (prefix + "_" + std::to_string(now) + ".h5");
}

void expect_events_equal(const std::vector<senna::core::domain::SpikeEvent>& lhs,
                         const std::vector<senna::core::domain::SpikeEvent>& rhs) {
    ASSERT_EQ(lhs.size(), rhs.size());
    for (std::size_t i = 0U; i < lhs.size(); ++i) {
        EXPECT_EQ(lhs[i].source, rhs[i].source);
        EXPECT_EQ(lhs[i].target, rhs[i].target);
        EXPECT_FLOAT_EQ(lhs[i].arrival, rhs[i].arrival);
        EXPECT_FLOAT_EQ(lhs[i].value, rhs[i].value);
    }
}

void expect_neuron_snapshot_equal(const senna::core::domain::NeuronSnapshot& lhs,
                                  const senna::core::domain::NeuronSnapshot& rhs) {
    EXPECT_EQ(lhs.id, rhs.id);
    EXPECT_EQ(lhs.position.x, rhs.position.x);
    EXPECT_EQ(lhs.position.y, rhs.position.y);
    EXPECT_EQ(lhs.position.z, rhs.position.z);
    EXPECT_EQ(lhs.type, rhs.type);
    EXPECT_FLOAT_EQ(lhs.config.v_rest, rhs.config.v_rest);
    EXPECT_FLOAT_EQ(lhs.config.v_reset, rhs.config.v_reset);
    EXPECT_FLOAT_EQ(lhs.config.tau_m, rhs.config.tau_m);
    EXPECT_FLOAT_EQ(lhs.config.t_ref, rhs.config.t_ref);
    EXPECT_FLOAT_EQ(lhs.config.theta_base, rhs.config.theta_base);
    EXPECT_FLOAT_EQ(lhs.potential, rhs.potential);
    EXPECT_FLOAT_EQ(lhs.threshold, rhs.threshold);
    EXPECT_FLOAT_EQ(lhs.last_update_time, rhs.last_update_time);
    EXPECT_FLOAT_EQ(lhs.last_spike_time, rhs.last_spike_time);
    EXPECT_FLOAT_EQ(lhs.average_rate, rhs.average_rate);
    EXPECT_EQ(lhs.in_refractory, rhs.in_refractory);
}

struct Runtime {
    std::vector<senna::core::domain::Neuron> neurons{};
    senna::core::domain::SynapseStore synapses{};
    senna::core::engine::EventQueue queue{};
    senna::core::engine::TimeManager time{0.5F, 0.0F};
    senna::core::engine::SimulationEngine engine;

    Runtime()
        : neurons(make_neurons()),
          synapses(make_synapses(neurons)),
          queue{},
          time(0.5F, 0.0F),
          engine(neurons, synapses, queue, time) {
        queue.push(senna::core::domain::SpikeEvent{0U, 0U, 0.0F, 1.1F});
        queue.push(senna::core::domain::SpikeEvent{0U, 0U, 1.5F, 1.1F});
    }

    static std::vector<senna::core::domain::Neuron> make_neurons() {
        using senna::core::domain::Coord3D;
        using senna::core::domain::Neuron;
        using senna::core::domain::NeuronConfig;
        using senna::core::domain::NeuronType;

        NeuronConfig config{};
        config.theta_base = 1.0F;
        config.t_ref = 0.0F;

        std::vector<Neuron> neurons{};
        neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, config);
        neurons.emplace_back(1U, Coord3D{1U, 0U, 0U}, NeuronType::Excitatory, config);
        return neurons;
    }

    static senna::core::domain::SynapseStore make_synapses(
        const std::vector<senna::core::domain::Neuron>& neurons) {
        using senna::core::domain::SynapseStore;

        SynapseStore synapses{neurons.size()};
        const auto& pre = neurons[0U];
        const auto& post = neurons[1U];
        synapses.connect(pre.id(), post.id(), pre.position(), post.position(), pre.type(), 1.1F,
                         0.5F);
        return synapses;
    }
};

std::vector<std::size_t> run_ticks(senna::core::engine::SimulationEngine& engine,
                                   const std::size_t tick_count) {
    std::vector<std::size_t> trace{};
    trace.reserve(tick_count);
    for (std::size_t i = 0U; i < tick_count; ++i) {
        static_cast<void>(engine.tick());
        trace.push_back(engine.emitted_last_tick());
    }
    return trace;
}

}  // namespace

TEST(HDF5WriterTest, SpikeTraceRoundTripIsBitwiseStable) {
    const auto path = make_temp_h5_path("senna_spike_trace");
    const std::string file = path.string();

    const std::vector<senna::core::domain::SpikeEvent> source{
        {1U, 2U, 0.5F, 0.7F},
        {2U, 4U, 1.0F, -0.2F},
        {5U, 8U, 1.5F, 1.1F},
    };

    senna::core::persistence::HDF5Writer writer{file};
    writer.write_spike_trace(3U, source);
    const auto restored = writer.read_spike_trace(3U);

    expect_events_equal(source, restored);
    static_cast<void>(std::filesystem::remove(path));
}

TEST(HDF5WriterTest, SnapshotRoundTripPreservesNeuronAndSynapseState) {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;
    using senna::core::domain::SynapseStore;

    const auto path = make_temp_h5_path("senna_snapshot");
    const std::string file = path.string();

    NeuronConfig config{};
    config.theta_base = 1.2F;
    config.t_ref = 0.0F;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, config);
    neurons.emplace_back(1U, Coord3D{1U, 1U, 0U}, NeuronType::Inhibitory, config);

    static_cast<void>(neurons[0U].receive_input(0.0F, 0.5F));
    static_cast<void>(neurons[1U].receive_input(0.0F, 1.3F));
    neurons[0U].set_threshold(1.4F);
    neurons[0U].set_average_rate(3.0F);

    SynapseStore synapses{2U};
    synapses.connect(neurons[0U].id(), neurons[1U].id(), neurons[0U].position(),
                     neurons[1U].position(), neurons[0U].type(), 0.5F, 1.0F);
    synapses.connect(neurons[1U].id(), neurons[0U].id(), neurons[1U].position(),
                     neurons[0U].position(), neurons[1U].type(), 0.2F, 1.0F);

    senna::core::persistence::HDF5Writer writer{file};
    writer.write_snapshot(1U, neurons, synapses);

    const auto restored = writer.read_snapshot(1U);
    ASSERT_EQ(restored.neurons.size(), neurons.size());
    ASSERT_EQ(restored.synapses.size(), synapses.size());

    for (std::size_t i = 0U; i < neurons.size(); ++i) {
        expect_neuron_snapshot_equal(neurons[i].snapshot(), restored.neurons[i]);
    }

    for (std::size_t i = 0U; i < synapses.size(); ++i) {
        const auto& lhs = synapses.at(static_cast<senna::core::domain::SynapseId>(i));
        const auto& rhs = restored.synapses[i];
        EXPECT_EQ(lhs.pre_id, rhs.pre_id);
        EXPECT_EQ(lhs.post_id, rhs.post_id);
        EXPECT_FLOAT_EQ(lhs.weight, rhs.weight);
        EXPECT_FLOAT_EQ(lhs.delay, rhs.delay);
        EXPECT_EQ(lhs.sign, rhs.sign);
    }

    static_cast<void>(std::filesystem::remove(path));
}

TEST(HDF5WriterTest, MetricsRoundTripPreservesNameAndValue) {
    const auto path = make_temp_h5_path("senna_metrics");
    const std::string file = path.string();

    senna::core::persistence::HDF5Writer writer{file};
    writer.write_metrics(2U, std::unordered_map<std::string, double>{
                                 {"accuracy", 0.91},
                                 {"loss", 0.12},
                                 {"spikes_per_tick", 7.0},
                             });

    auto restored = writer.read_metrics(2U);
    std::sort(restored.begin(), restored.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.name < rhs.name; });

    ASSERT_EQ(restored.size(), 3U);
    EXPECT_EQ(restored[0U].name, "accuracy");
    EXPECT_DOUBLE_EQ(restored[0U].value, 0.91);
    EXPECT_EQ(restored[1U].name, "loss");
    EXPECT_DOUBLE_EQ(restored[1U].value, 0.12);
    EXPECT_EQ(restored[2U].name, "spikes_per_tick");
    EXPECT_DOUBLE_EQ(restored[2U].value, 7.0);

    static_cast<void>(std::filesystem::remove(path));
}

TEST(StateSerializerTest, SaveLoadAndContinuationMatchReferenceRun) {
    const auto path = make_temp_h5_path("senna_state");
    const std::string file = path.string();

    Runtime reference{};
    const auto reference_trace = run_ticks(reference.engine, 8U);

    Runtime split{};
    auto split_trace = run_ticks(split.engine, 4U);

    const auto captured =
        senna::core::persistence::StateSerializer::capture(split.neurons, split.synapses,
                                                           split.queue, split.time, 42U);
    senna::core::persistence::StateSerializer::save_state(file, captured);

    const auto loaded = senna::core::persistence::StateSerializer::load_state(file);
    EXPECT_EQ(loaded.rng_state, 42U);

    std::vector<senna::core::domain::Neuron> restored_neurons{};
    senna::core::domain::SynapseStore restored_synapses{};
    senna::core::engine::EventQueue restored_queue{};
    senna::core::persistence::StateSerializer::restore(loaded, restored_neurons,
                                                       restored_synapses, restored_queue);
    auto restored_time = senna::core::persistence::StateSerializer::make_time_manager(loaded);

    senna::core::engine::SimulationEngine restored_engine{restored_neurons, restored_synapses,
                                                           restored_queue, restored_time};

    const auto tail_trace = run_ticks(restored_engine, 4U);
    split_trace.insert(split_trace.end(), tail_trace.begin(), tail_trace.end());

    EXPECT_EQ(reference_trace, split_trace);
    static_cast<void>(std::filesystem::remove(path));
}

TEST(EpochArtifactPipelineTest, WritesEpochFileToOutboxAutomatically) {
    const auto root = std::filesystem::temp_directory_path() /
                      ("senna_artifacts_" +
                       std::to_string(std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count()));

    const auto experiment_file = root / "experiment.h5";
    const auto outbox_dir = root / "outbox";

    Runtime runtime{};
    const auto trace = run_ticks(runtime.engine, 4U);
    const std::vector<senna::core::domain::SpikeEvent> emitted =
        runtime.engine.emitted_events_last_tick();

    senna::core::persistence::EpochArtifactConfig config{};
    config.experiment_file = experiment_file;
    config.outbox_dir = outbox_dir;
    config.include_state_snapshot = true;

    const auto state = senna::core::persistence::StateSerializer::capture(
        runtime.neurons, runtime.synapses, runtime.queue, runtime.time, 7U);

    senna::core::persistence::EpochArtifactPipeline pipeline{config};
    pipeline.persist_epoch(
        2U, emitted, runtime.neurons, runtime.synapses,
        std::unordered_map<std::string, double>{
            {"trace_len", static_cast<double>(trace.size())},
            {"emitted_last_tick", static_cast<double>(runtime.engine.emitted_last_tick())},
        },
        &state);

    ASSERT_TRUE(std::filesystem::exists(experiment_file));

    const auto outbox_epoch_file = pipeline.outbox_file_path(2U);
    EXPECT_EQ(outbox_epoch_file.filename().string(), "epoch_000000002.h5");
    ASSERT_TRUE(std::filesystem::exists(outbox_epoch_file));

    senna::core::persistence::HDF5Writer outbox_reader{outbox_epoch_file.string()};
    const auto restored_metrics = outbox_reader.read_metrics(2U);
    ASSERT_EQ(restored_metrics.size(), 2U);

    const auto restored_state =
        senna::core::persistence::StateSerializer::load_state(outbox_epoch_file.string());
    EXPECT_EQ(restored_state.rng_state, 7U);
    EXPECT_EQ(restored_state.neurons.size(), runtime.neurons.size());
    EXPECT_EQ(restored_state.synapses.size(), runtime.synapses.size());

    static_cast<void>(std::filesystem::remove_all(root));
}

#pragma once

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/domain/types.h"
#include "core/engine/event_queue.h"
#include "core/engine/time_manager.h"
#include "core/persistence/hdf5_writer.h"
#include "core/persistence/state_serializer.h"

namespace senna::core::persistence {

struct EpochArtifactConfig {
    std::filesystem::path experiment_file{"data/artifacts/experiment.h5"};
    std::filesystem::path outbox_dir{"data/artifacts/outbox"};
    bool include_state_snapshot{true};
};

class EpochArtifactPipeline final {
   public:
    explicit EpochArtifactPipeline(EpochArtifactConfig config = {})
        : config_(std::move(config)),
          experiment_writer_(config_.experiment_file.string()) {
        ensure_directories();
    }

    void persist_epoch(const std::size_t epoch,
                       const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                       const std::vector<senna::core::domain::Neuron>& neurons,
                       const senna::core::domain::SynapseStore& synapses,
                       const std::vector<MetricPoint>& metrics,
                       const SimulationState* state = nullptr) const {
        experiment_writer_.write_spike_trace(epoch, spike_trace);
        experiment_writer_.write_snapshot(epoch, neurons, synapses);
        experiment_writer_.write_metrics(epoch, metrics);

        write_outbox_epoch_file(epoch, spike_trace, neurons, synapses, metrics, state);
    }

    void persist_epoch(const std::size_t epoch,
                       const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                       const std::vector<senna::core::domain::Neuron>& neurons,
                       const senna::core::domain::SynapseStore& synapses,
                       const std::unordered_map<std::string, double>& metrics,
                       const SimulationState* state = nullptr) const {
        std::vector<MetricPoint> ordered_metrics{};
        ordered_metrics.reserve(metrics.size());
        for (const auto& [name, value] : metrics) {
            ordered_metrics.push_back(MetricPoint{name, value});
        }

        persist_epoch(epoch, spike_trace, neurons, synapses, ordered_metrics, state);
    }

    void persist_epoch_with_capture(const std::size_t epoch,
                                    const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                                    const std::vector<senna::core::domain::Neuron>& neurons,
                                    const senna::core::domain::SynapseStore& synapses,
                                    const std::unordered_map<std::string, double>& metrics,
                                    const senna::core::engine::EventQueue& queue,
                                    const senna::core::engine::TimeManager& time,
                                    const std::uint64_t rng_state = 0U) const {
        const auto state = StateSerializer::capture(neurons, synapses, queue, time, rng_state);
        persist_epoch(epoch, spike_trace, neurons, synapses, metrics, &state);
    }

    [[nodiscard]] const EpochArtifactConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::filesystem::path outbox_file_path(const std::size_t epoch) const {
        char file_name[32]{};
        std::snprintf(file_name, sizeof(file_name), "epoch_%09zu.h5", epoch);
        return config_.outbox_dir / file_name;
    }

   private:
    void ensure_directories() const {
        const auto experiment_parent = config_.experiment_file.parent_path();
        if (!experiment_parent.empty()) {
            std::filesystem::create_directories(experiment_parent);
        }

        if (config_.outbox_dir.empty()) {
            throw std::invalid_argument("EpochArtifactConfig.outbox_dir must not be empty");
        }
        std::filesystem::create_directories(config_.outbox_dir);
    }

    void write_outbox_epoch_file(const std::size_t epoch,
                                 const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                                 const std::vector<senna::core::domain::Neuron>& neurons,
                                 const senna::core::domain::SynapseStore& synapses,
                                 const std::vector<MetricPoint>& metrics,
                                 const SimulationState* state) const {
        const auto destination = outbox_file_path(epoch);
        const auto temporary = destination.string() + ".tmp";

        std::filesystem::remove(destination);
        std::filesystem::remove(temporary);

        HDF5Writer outbox_writer{temporary};
        outbox_writer.write_spike_trace(epoch, spike_trace);
        outbox_writer.write_snapshot(epoch, neurons, synapses);
        outbox_writer.write_metrics(epoch, metrics);

        if (config_.include_state_snapshot && state != nullptr) {
            StateSerializer::save_state(temporary, *state);
        }

        std::filesystem::rename(temporary, destination);
    }

    EpochArtifactConfig config_{};
    HDF5Writer experiment_writer_;
};

}  // namespace senna::core::persistence

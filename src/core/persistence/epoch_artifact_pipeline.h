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
    explicit EpochArtifactPipeline(EpochArtifactConfig config = {});

    void persist_epoch(std::size_t epoch,
                       const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                       const std::vector<senna::core::domain::Neuron>& neurons,
                       const senna::core::domain::SynapseStore& synapses,
                       const std::vector<MetricPoint>& metrics,
                       const SimulationState* state = nullptr) const;

    void persist_epoch(std::size_t epoch,
                       const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                       const std::vector<senna::core::domain::Neuron>& neurons,
                       const senna::core::domain::SynapseStore& synapses,
                       const std::unordered_map<std::string, double>& metrics,
                       const SimulationState* state = nullptr) const;

    void persist_epoch_with_capture(std::size_t epoch,
                                    const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                                    const std::vector<senna::core::domain::Neuron>& neurons,
                                    const senna::core::domain::SynapseStore& synapses,
                                    const std::unordered_map<std::string, double>& metrics,
                                    const senna::core::engine::EventQueue& queue,
                                    const senna::core::engine::TimeManager& time,
                                    std::uint64_t rng_state = 0U) const;

    [[nodiscard]] const EpochArtifactConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::filesystem::path outbox_file_path(std::size_t epoch) const;

   private:
    void ensure_directories() const;

    void write_outbox_epoch_file(std::size_t epoch,
                                 const std::vector<senna::core::domain::SpikeEvent>& spike_trace,
                                 const std::vector<senna::core::domain::Neuron>& neurons,
                                 const senna::core::domain::SynapseStore& synapses,
                                 const std::vector<MetricPoint>& metrics,
                                 const SimulationState* state) const;

    EpochArtifactConfig config_{};
    HDF5Writer experiment_writer_;
};

}  // namespace senna::core::persistence

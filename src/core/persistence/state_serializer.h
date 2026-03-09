#pragma once

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/engine/event_queue.h"
#include "core/engine/time_manager.h"
#include "core/persistence/hdf5_writer.h"

namespace senna::core::persistence {

struct SimulationState {
    std::vector<senna::core::domain::NeuronSnapshot> neurons{};
    std::vector<senna::core::domain::Synapse> synapses{};
    std::vector<senna::core::domain::SpikeEvent> pending_events{};
    senna::core::domain::Time elapsed{0.0F};
    senna::core::domain::Time dt{0.5F};
    std::uint64_t rng_state{0U};
};

namespace detail {

struct StateMetadataRecord {
    float elapsed{};
    float dt{};
    std::uint64_t rng_state{};
};

[[nodiscard]] hid_t make_state_metadata_type();

}  // namespace detail

class StateSerializer final {
   public:
    static void save_state(const std::string& path, const SimulationState& state);

    [[nodiscard]] static SimulationState load_state(const std::string& path);

    [[nodiscard]] static SimulationState capture(
        const std::vector<senna::core::domain::Neuron>& neurons,
        const senna::core::domain::SynapseStore& synapses,
        const senna::core::engine::EventQueue& queue, const senna::core::engine::TimeManager& time,
        std::uint64_t rng_state = 0U);

    static void restore(const SimulationState& state,
                        std::vector<senna::core::domain::Neuron>& neurons,
                        senna::core::domain::SynapseStore& synapses,
                        senna::core::engine::EventQueue& queue);

    [[nodiscard]] static senna::core::engine::TimeManager make_time_manager(
        const SimulationState& state);
};

}  // namespace senna::core::persistence

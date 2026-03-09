#include "core/persistence/state_serializer.h"

namespace senna::core::persistence {

namespace detail {

hid_t make_state_metadata_type() {
    const auto type = check_id(H5Tcreate(H5T_COMPOUND, sizeof(StateMetadataRecord)),
                               "Failed to create StateMetadata HDF5 type");

    check_status(
        H5Tinsert(type, "elapsed", HOFFSET(StateMetadataRecord, elapsed), H5T_NATIVE_FLOAT),
        "Failed to insert StateMetadata.elapsed");
    check_status(H5Tinsert(type, "dt", HOFFSET(StateMetadataRecord, dt), H5T_NATIVE_FLOAT),
                 "Failed to insert StateMetadata.dt");
    check_status(
        H5Tinsert(type, "rng_state", HOFFSET(StateMetadataRecord, rng_state), H5T_NATIVE_UINT64),
        "Failed to insert StateMetadata.rng_state");

    return type;
}

}  // namespace detail

void StateSerializer::save_state(const std::string& path, const SimulationState& state) {
    auto file = detail::open_rw_or_create_file(path);
    auto state_group = detail::open_or_create_group_path(file.id, "/state");

    auto neuron_type =
        detail::ScopedH5{detail::make_neuron_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto synapse_type =
        detail::ScopedH5{detail::make_synapse_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto spike_type =
        detail::ScopedH5{detail::make_spike_event_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto metadata_type = detail::ScopedH5{detail::make_state_metadata_type(),
                                          static_cast<herr_t (*)(hid_t)>(H5Tclose)};

    std::vector<detail::NeuronRecord> neuron_records{};
    neuron_records.reserve(state.neurons.size());
    for (const auto& neuron : state.neurons) {
        neuron_records.push_back(detail::to_record(neuron));
    }

    std::vector<detail::SynapseRecord> synapse_records{};
    synapse_records.reserve(state.synapses.size());
    for (const auto& synapse : state.synapses) {
        synapse_records.push_back(detail::to_record(synapse));
    }

    std::vector<detail::SpikeEventRecord> pending_records{};
    pending_records.reserve(state.pending_events.size());
    for (const auto& event : state.pending_events) {
        pending_records.push_back(detail::to_record(event));
    }

    detail::write_compound_dataset(state_group.id, "neurons", neuron_type.id, neuron_records);
    detail::write_compound_dataset(state_group.id, "synapses", synapse_type.id, synapse_records);
    detail::write_compound_dataset(state_group.id, "pending_events", spike_type.id,
                                   pending_records);

    const std::vector<detail::StateMetadataRecord> metadata{
        detail::StateMetadataRecord{state.elapsed, state.dt, state.rng_state}};
    detail::write_compound_dataset(state_group.id, "metadata", metadata_type.id, metadata);
}

SimulationState StateSerializer::load_state(const std::string& path) {
    auto file = detail::open_ro_file(path);
    auto state_group = detail::open_group_path(file.id, "/state");

    auto neuron_type =
        detail::ScopedH5{detail::make_neuron_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto synapse_type =
        detail::ScopedH5{detail::make_synapse_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto spike_type =
        detail::ScopedH5{detail::make_spike_event_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    auto metadata_type = detail::ScopedH5{detail::make_state_metadata_type(),
                                          static_cast<herr_t (*)(hid_t)>(H5Tclose)};

    const auto neuron_records = detail::read_compound_dataset<detail::NeuronRecord>(
        state_group.id, "neurons", neuron_type.id);
    const auto synapse_records = detail::read_compound_dataset<detail::SynapseRecord>(
        state_group.id, "synapses", synapse_type.id);
    const auto pending_records = detail::read_compound_dataset<detail::SpikeEventRecord>(
        state_group.id, "pending_events", spike_type.id);
    const auto metadata_records = detail::read_compound_dataset<detail::StateMetadataRecord>(
        state_group.id, "metadata", metadata_type.id);

    if (metadata_records.size() != 1U) {
        throw std::runtime_error("State metadata dataset must contain exactly one record");
    }

    SimulationState state{};
    state.neurons.reserve(neuron_records.size());
    for (const auto& neuron_record : neuron_records) {
        state.neurons.push_back(detail::from_record(neuron_record));
    }

    state.synapses.reserve(synapse_records.size());
    for (const auto& synapse_record : synapse_records) {
        state.synapses.push_back(detail::from_record(synapse_record));
    }

    state.pending_events.reserve(pending_records.size());
    for (const auto& event_record : pending_records) {
        state.pending_events.push_back(detail::from_record(event_record));
    }

    state.elapsed = metadata_records.front().elapsed;
    state.dt = metadata_records.front().dt;
    state.rng_state = metadata_records.front().rng_state;

    return state;
}

SimulationState StateSerializer::capture(const std::vector<senna::core::domain::Neuron>& neurons,
                                         const senna::core::domain::SynapseStore& synapses,
                                         const senna::core::engine::EventQueue& queue,
                                         const senna::core::engine::TimeManager& time,
                                         const std::uint64_t rng_state) {
    SimulationState state{};
    state.neurons.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        state.neurons.push_back(neuron.snapshot());
    }

    state.synapses = synapses.synapses();
    state.pending_events = queue.snapshot();
    state.elapsed = time.elapsed();
    state.dt = time.dt();
    state.rng_state = rng_state;
    return state;
}

void StateSerializer::restore(const SimulationState& state,
                              std::vector<senna::core::domain::Neuron>& neurons,
                              senna::core::domain::SynapseStore& synapses,
                              senna::core::engine::EventQueue& queue) {
    neurons.clear();
    neurons.reserve(state.neurons.size());
    for (const auto& neuron_state : state.neurons) {
        neurons.push_back(senna::core::domain::Neuron::from_snapshot(neuron_state));
    }

    synapses.synapses() = state.synapses;
    synapses.rebuild_indices(state.neurons.size());
    queue.restore(state.pending_events);
}

senna::core::engine::TimeManager StateSerializer::make_time_manager(const SimulationState& state) {
    return senna::core::engine::TimeManager{state.dt, state.elapsed};
}

}  // namespace senna::core::persistence

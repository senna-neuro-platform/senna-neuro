#include "core/persistence/hdf5_writer.h"

namespace senna::core::persistence {

namespace detail {

void check_status(const herr_t status, const char* message) {
    if (status < 0) {
        throw std::runtime_error(message);
    }
}

hid_t check_id(const hid_t id, const char* message) {
    if (id < 0) {
        throw std::runtime_error(message);
    }
    return id;
}

bool link_exists(const hid_t parent_id, const std::string& name) {
    return H5Lexists(parent_id, name.c_str(), H5P_DEFAULT) > 0;
}

ScopedH5 open_rw_or_create_file(const std::string& path) {
    if (std::filesystem::exists(path)) {
        return ScopedH5{check_id(H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT),
                                 "Failed to open HDF5 file for read/write"),
                        H5Fclose};
    }

    return ScopedH5{check_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                             "Failed to create HDF5 file"),
                    H5Fclose};
}

ScopedH5 open_ro_file(const std::string& path) {
    return ScopedH5{
        check_id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
                 "Failed to open HDF5 file for read-only access"),
        H5Fclose,
    };
}

std::vector<std::string> split_path(const std::string& path) {
    std::vector<std::string> parts{};
    std::string current{};

    for (const char ch : path) {
        if (ch == '/') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }

    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

ScopedH5 open_or_create_group(const hid_t parent_id, const std::string& name) {
    if (link_exists(parent_id, name)) {
        return ScopedH5{
            check_id(H5Gopen2(parent_id, name.c_str(), H5P_DEFAULT), "Failed to open HDF5 group"),
            H5Gclose};
    }

    return ScopedH5{
        check_id(H5Gcreate2(parent_id, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                 "Failed to create HDF5 group"),
        H5Gclose,
    };
}

ScopedH5 open_group(const hid_t parent_id, const std::string& name) {
    return ScopedH5{
        check_id(H5Gopen2(parent_id, name.c_str(), H5P_DEFAULT), "Failed to open HDF5 group"),
        H5Gclose};
}

ScopedH5 open_or_create_group_path(const hid_t root_id, const std::string& path) {
    auto current = ScopedH5{root_id, nullptr};

    for (const auto& part : split_path(path)) {
        auto next = open_or_create_group(current.id, part);
        current = std::move(next);
    }

    return current;
}

ScopedH5 open_group_path(const hid_t root_id, const std::string& path) {
    auto current = ScopedH5{root_id, nullptr};

    for (const auto& part : split_path(path)) {
        auto next = open_group(current.id, part);
        current = std::move(next);
    }

    return current;
}

void delete_dataset_if_exists(const hid_t group_id, const std::string& dataset_name) {
    if (link_exists(group_id, dataset_name)) {
        check_status(H5Ldelete(group_id, dataset_name.c_str(), H5P_DEFAULT),
                     "Failed to delete existing HDF5 dataset");
    }
}

hid_t make_spike_event_type() {
    const auto type = check_id(H5Tcreate(H5T_COMPOUND, sizeof(SpikeEventRecord)),
                               "Failed to create SpikeEvent HDF5 type");

    check_status(H5Tinsert(type, "source", HOFFSET(SpikeEventRecord, source), H5T_NATIVE_UINT32),
                 "Failed to insert SpikeEvent.source field");
    check_status(H5Tinsert(type, "target", HOFFSET(SpikeEventRecord, target), H5T_NATIVE_UINT32),
                 "Failed to insert SpikeEvent.target field");
    check_status(H5Tinsert(type, "arrival", HOFFSET(SpikeEventRecord, arrival), H5T_NATIVE_FLOAT),
                 "Failed to insert SpikeEvent.arrival field");
    check_status(H5Tinsert(type, "value", HOFFSET(SpikeEventRecord, value), H5T_NATIVE_FLOAT),
                 "Failed to insert SpikeEvent.value field");

    return type;
}

hid_t make_synapse_type() {
    const auto type =
        check_id(H5Tcreate(H5T_COMPOUND, sizeof(SynapseRecord)), "Failed to create Synapse type");

    check_status(H5Tinsert(type, "pre_id", HOFFSET(SynapseRecord, pre_id), H5T_NATIVE_UINT32),
                 "Failed to insert Synapse.pre_id");
    check_status(H5Tinsert(type, "post_id", HOFFSET(SynapseRecord, post_id), H5T_NATIVE_UINT32),
                 "Failed to insert Synapse.post_id");
    check_status(H5Tinsert(type, "weight", HOFFSET(SynapseRecord, weight), H5T_NATIVE_FLOAT),
                 "Failed to insert Synapse.weight");
    check_status(H5Tinsert(type, "delay", HOFFSET(SynapseRecord, delay), H5T_NATIVE_FLOAT),
                 "Failed to insert Synapse.delay");
    check_status(H5Tinsert(type, "sign", HOFFSET(SynapseRecord, sign), H5T_NATIVE_INT8),
                 "Failed to insert Synapse.sign");

    return type;
}

hid_t make_neuron_type() {
    const auto type =
        check_id(H5Tcreate(H5T_COMPOUND, sizeof(NeuronRecord)), "Failed to create Neuron type");

    check_status(H5Tinsert(type, "id", HOFFSET(NeuronRecord, id), H5T_NATIVE_UINT32),
                 "Failed to insert Neuron.id");
    check_status(H5Tinsert(type, "x", HOFFSET(NeuronRecord, x), H5T_NATIVE_UINT16),
                 "Failed to insert Neuron.x");
    check_status(H5Tinsert(type, "y", HOFFSET(NeuronRecord, y), H5T_NATIVE_UINT16),
                 "Failed to insert Neuron.y");
    check_status(H5Tinsert(type, "z", HOFFSET(NeuronRecord, z), H5T_NATIVE_UINT16),
                 "Failed to insert Neuron.z");
    check_status(H5Tinsert(type, "type", HOFFSET(NeuronRecord, type), H5T_NATIVE_UINT8),
                 "Failed to insert Neuron.type");

    check_status(H5Tinsert(type, "v_rest", HOFFSET(NeuronRecord, v_rest), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.v_rest");
    check_status(H5Tinsert(type, "v_reset", HOFFSET(NeuronRecord, v_reset), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.v_reset");
    check_status(H5Tinsert(type, "tau_m", HOFFSET(NeuronRecord, tau_m), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.tau_m");
    check_status(H5Tinsert(type, "t_ref", HOFFSET(NeuronRecord, t_ref), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.t_ref");
    check_status(H5Tinsert(type, "theta_base", HOFFSET(NeuronRecord, theta_base), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.theta_base");

    check_status(H5Tinsert(type, "potential", HOFFSET(NeuronRecord, potential), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.potential");
    check_status(H5Tinsert(type, "threshold", HOFFSET(NeuronRecord, threshold), H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.threshold");
    check_status(H5Tinsert(type, "last_update_time", HOFFSET(NeuronRecord, last_update_time),
                           H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.last_update_time");
    check_status(H5Tinsert(type, "last_spike_time", HOFFSET(NeuronRecord, last_spike_time),
                           H5T_NATIVE_FLOAT),
                 "Failed to insert Neuron.last_spike_time");
    check_status(
        H5Tinsert(type, "average_rate", HOFFSET(NeuronRecord, average_rate), H5T_NATIVE_FLOAT),
        "Failed to insert Neuron.average_rate");
    check_status(
        H5Tinsert(type, "in_refractory", HOFFSET(NeuronRecord, in_refractory), H5T_NATIVE_UINT8),
        "Failed to insert Neuron.in_refractory");

    return type;
}

hid_t make_metric_type() {
    const auto type =
        check_id(H5Tcreate(H5T_COMPOUND, sizeof(MetricRecord)), "Failed to create Metric type");
    const auto str_type =
        check_id(H5Tcopy(H5T_C_S1), "Failed to allocate HDF5 string type for Metric.name");

    check_status(H5Tset_size(str_type, kMetricNameBytes),
                 "Failed to set HDF5 string size for Metric.name");
    check_status(H5Tset_strpad(str_type, H5T_STR_NULLTERM),
                 "Failed to set HDF5 string padding for Metric.name");

    check_status(H5Tinsert(type, "name", HOFFSET(MetricRecord, name), str_type),
                 "Failed to insert Metric.name");
    check_status(H5Tinsert(type, "value", HOFFSET(MetricRecord, value), H5T_NATIVE_DOUBLE),
                 "Failed to insert Metric.value");

    check_status(H5Tclose(str_type), "Failed to close temporary HDF5 string type");
    return type;
}

SpikeEventRecord to_record(const senna::core::domain::SpikeEvent& event) {
    return SpikeEventRecord{event.source, event.target, event.arrival, event.value};
}

senna::core::domain::SpikeEvent from_record(const SpikeEventRecord& record) {
    return senna::core::domain::SpikeEvent{record.source, record.target, record.arrival,
                                           record.value};
}

SynapseRecord to_record(const senna::core::domain::Synapse& synapse) {
    return SynapseRecord{synapse.pre_id, synapse.post_id, synapse.weight, synapse.delay,
                         synapse.sign};
}

senna::core::domain::Synapse from_record(const SynapseRecord& record) {
    return senna::core::domain::Synapse{record.pre_id, record.post_id, record.weight, record.delay,
                                        record.sign};
}

NeuronRecord to_record(const senna::core::domain::NeuronSnapshot& neuron) {
    return NeuronRecord{
        neuron.id,
        neuron.position.x,
        neuron.position.y,
        neuron.position.z,
        static_cast<std::uint8_t>(neuron.type),
        neuron.config.v_rest,
        neuron.config.v_reset,
        neuron.config.tau_m,
        neuron.config.t_ref,
        neuron.config.theta_base,
        neuron.potential,
        neuron.threshold,
        neuron.last_update_time,
        neuron.last_spike_time,
        neuron.average_rate,
        static_cast<std::uint8_t>(neuron.in_refractory ? 1U : 0U),
    };
}

senna::core::domain::NeuronSnapshot from_record(const NeuronRecord& record) {
    return senna::core::domain::NeuronSnapshot{
        record.id,
        senna::core::domain::Coord3D{record.x, record.y, record.z},
        record.type == 0U ? senna::core::domain::NeuronType::Excitatory
                          : senna::core::domain::NeuronType::Inhibitory,
        senna::core::domain::NeuronConfig{record.v_rest, record.v_reset, record.tau_m, record.t_ref,
                                          record.theta_base},
        record.potential,
        record.threshold,
        record.last_update_time,
        record.last_spike_time,
        record.average_rate,
        record.in_refractory != 0U,
    };
}

MetricRecord to_record(const MetricPoint& metric) {
    MetricRecord record{};
    std::snprintf(record.name, sizeof(record.name), "%s", metric.name.c_str());
    record.value = metric.value;
    return record;
}

MetricPoint from_record(const MetricRecord& record) {
    return MetricPoint{record.name, record.value};
}

}  // namespace detail

HDF5Writer::HDF5Writer(std::string file_path) : file_path_(std::move(file_path)) {}

void HDF5Writer::write_epoch(const std::size_t epoch,
                             const std::vector<senna::core::domain::SpikeEvent>& trace,
                             const std::vector<senna::core::domain::Neuron>& neurons,
                             const senna::core::domain::SynapseStore& synapses,
                             const std::vector<MetricPoint>& metrics) const {
    write_epoch(epoch, trace, snapshot_buffer_from(neurons), synapses.synapses(), metrics);
}

void HDF5Writer::write_epoch(const std::size_t epoch,
                             const std::vector<senna::core::domain::SpikeEvent>& trace,
                             const std::vector<senna::core::domain::NeuronSnapshot>& neurons,
                             const std::vector<senna::core::domain::Synapse>& synapses,
                             const std::vector<MetricPoint>& metrics) const {
    auto file = detail::open_rw_or_create_file(file_path_);
    auto epoch_group = detail::open_or_create_group_path(file.id, epoch_group_path(epoch));
    write_spike_trace_group(epoch_group.id, trace);
    write_snapshot_group(epoch_group.id, neurons, synapses);
    write_metrics_group(epoch_group.id, metrics);
}

void HDF5Writer::write_spike_trace(
    const std::size_t epoch, const std::vector<senna::core::domain::SpikeEvent>& trace) const {
    auto file = detail::open_rw_or_create_file(file_path_);
    auto epoch_group = detail::open_or_create_group_path(file.id, epoch_group_path(epoch));
    write_spike_trace_group(epoch_group.id, trace);
}

void HDF5Writer::write_snapshot(const std::size_t epoch,
                                const std::vector<senna::core::domain::Neuron>& neurons,
                                const senna::core::domain::SynapseStore& synapses) const {
    write_snapshot(epoch, snapshot_buffer_from(neurons), synapses.synapses());
}

void HDF5Writer::write_snapshot(const std::size_t epoch,
                                const std::vector<senna::core::domain::NeuronSnapshot>& neurons,
                                const std::vector<senna::core::domain::Synapse>& synapses) const {
    auto file = detail::open_rw_or_create_file(file_path_);
    auto epoch_group = detail::open_or_create_group_path(file.id, epoch_group_path(epoch));
    write_snapshot_group(epoch_group.id, neurons, synapses);
}

void HDF5Writer::write_metrics(const std::size_t epoch,
                               const std::vector<MetricPoint>& metrics) const {
    auto file = detail::open_rw_or_create_file(file_path_);
    auto epoch_group = detail::open_or_create_group_path(file.id, epoch_group_path(epoch));
    write_metrics_group(epoch_group.id, metrics);
}

void HDF5Writer::write_metrics(const std::size_t epoch,
                               const std::unordered_map<std::string, double>& metrics) const {
    write_metrics(epoch, sorted_metric_buffer_from(metrics));
}

std::vector<senna::core::domain::SpikeEvent> HDF5Writer::read_spike_trace(
    const std::size_t epoch) const {
    auto file = detail::open_ro_file(file_path_);
    auto epoch_group = detail::open_group_path(file.id, epoch_group_path(epoch));

    const auto records = detail::read_compound_dataset<detail::SpikeEventRecord>(
        epoch_group.id, "spike_trace", ensure_spike_type());

    std::vector<senna::core::domain::SpikeEvent> trace{};
    trace.reserve(records.size());
    for (const auto& record : records) {
        trace.push_back(detail::from_record(record));
    }
    return trace;
}

SnapshotData HDF5Writer::read_snapshot(const std::size_t epoch) const {
    auto file = detail::open_ro_file(file_path_);
    auto epoch_group = detail::open_group_path(file.id, epoch_group_path(epoch));

    const auto neuron_records = detail::read_compound_dataset<detail::NeuronRecord>(
        epoch_group.id, "neurons", ensure_neuron_type());
    const auto synapse_records = detail::read_compound_dataset<detail::SynapseRecord>(
        epoch_group.id, "synapses", ensure_synapse_type());

    SnapshotData data{};
    data.neurons.reserve(neuron_records.size());
    for (const auto& record : neuron_records) {
        data.neurons.push_back(detail::from_record(record));
    }

    data.synapses.reserve(synapse_records.size());
    for (const auto& record : synapse_records) {
        data.synapses.push_back(detail::from_record(record));
    }

    return data;
}

std::vector<MetricPoint> HDF5Writer::read_metrics(const std::size_t epoch) const {
    auto file = detail::open_ro_file(file_path_);
    auto epoch_group = detail::open_group_path(file.id, epoch_group_path(epoch));

    const auto records = detail::read_compound_dataset<detail::MetricRecord>(
        epoch_group.id, "metrics", ensure_metric_type());

    std::vector<MetricPoint> metrics{};
    metrics.reserve(records.size());
    for (const auto& record : records) {
        metrics.push_back(detail::from_record(record));
    }
    return metrics;
}

hid_t HDF5Writer::ensure_spike_type() const {
    if (spike_type_.id < 0) {
        spike_type_ = detail::ScopedH5{detail::make_spike_event_type(),
                                       static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    }
    return spike_type_.id;
}

hid_t HDF5Writer::ensure_synapse_type() const {
    if (synapse_type_.id < 0) {
        synapse_type_ =
            detail::ScopedH5{detail::make_synapse_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    }
    return synapse_type_.id;
}

hid_t HDF5Writer::ensure_neuron_type() const {
    if (neuron_type_.id < 0) {
        neuron_type_ =
            detail::ScopedH5{detail::make_neuron_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    }
    return neuron_type_.id;
}

hid_t HDF5Writer::ensure_metric_type() const {
    if (metric_type_.id < 0) {
        metric_type_ =
            detail::ScopedH5{detail::make_metric_type(), static_cast<herr_t (*)(hid_t)>(H5Tclose)};
    }
    return metric_type_.id;
}

void HDF5Writer::write_spike_trace_group(
    const hid_t epoch_group_id, const std::vector<senna::core::domain::SpikeEvent>& trace) const {
    spike_record_buffer_.clear();
    spike_record_buffer_.reserve(trace.size());
    for (const auto& event : trace) {
        spike_record_buffer_.push_back(detail::to_record(event));
    }

    detail::write_compound_dataset(epoch_group_id, "spike_trace", ensure_spike_type(),
                                   spike_record_buffer_);
}

void HDF5Writer::write_snapshot_group(
    const hid_t epoch_group_id, const std::vector<senna::core::domain::NeuronSnapshot>& neurons,
    const std::vector<senna::core::domain::Synapse>& synapses) const {
    neuron_record_buffer_.clear();
    neuron_record_buffer_.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        neuron_record_buffer_.push_back(detail::to_record(neuron));
    }

    synapse_record_buffer_.clear();
    synapse_record_buffer_.reserve(synapses.size());
    for (const auto& synapse : synapses) {
        synapse_record_buffer_.push_back(detail::to_record(synapse));
    }

    detail::write_compound_dataset(epoch_group_id, "neurons", ensure_neuron_type(),
                                   neuron_record_buffer_);
    detail::write_compound_dataset(epoch_group_id, "synapses", ensure_synapse_type(),
                                   synapse_record_buffer_);
}

void HDF5Writer::write_metrics_group(const hid_t epoch_group_id,
                                     const std::vector<MetricPoint>& metrics) const {
    metric_record_buffer_.clear();
    metric_record_buffer_.reserve(metrics.size());
    for (const auto& metric : metrics) {
        metric_record_buffer_.push_back(detail::to_record(metric));
    }

    detail::write_compound_dataset(epoch_group_id, "metrics", ensure_metric_type(),
                                   metric_record_buffer_);
}

const std::vector<senna::core::domain::NeuronSnapshot>& HDF5Writer::snapshot_buffer_from(
    const std::vector<senna::core::domain::Neuron>& neurons) const {
    neuron_snapshot_buffer_.clear();
    neuron_snapshot_buffer_.reserve(neurons.size());
    for (const auto& neuron : neurons) {
        neuron_snapshot_buffer_.push_back(neuron.snapshot());
    }
    return neuron_snapshot_buffer_;
}

const std::vector<MetricPoint>& HDF5Writer::sorted_metric_buffer_from(
    const std::unordered_map<std::string, double>& metrics) const {
    metric_point_buffer_.clear();
    metric_point_buffer_.reserve(metrics.size());
    for (const auto& [name, value] : metrics) {
        metric_point_buffer_.push_back(MetricPoint{name, value});
    }

    std::sort(metric_point_buffer_.begin(), metric_point_buffer_.end(),
              [](const MetricPoint& lhs, const MetricPoint& rhs) { return lhs.name < rhs.name; });
    return metric_point_buffer_;
}

std::string HDF5Writer::epoch_group_path(const std::size_t epoch) {
    char epoch_name[32]{};
    std::snprintf(epoch_name, sizeof(epoch_name), "epoch_%06zu", epoch);
    return std::string{"/epochs/"} + epoch_name;
}

}  // namespace senna::core::persistence

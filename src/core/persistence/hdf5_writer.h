#pragma once

#include <hdf5.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/synapse.h"
#include "core/domain/types.h"

namespace senna::core::persistence {

struct SnapshotData {
    std::vector<senna::core::domain::NeuronSnapshot> neurons{};
    std::vector<senna::core::domain::Synapse> synapses{};
};

struct MetricPoint {
    std::string name{};
    double value{0.0};
};

namespace detail {

constexpr std::size_t kMetricNameBytes = 64U;

struct ScopedH5 {
    hid_t id{-1};
    herr_t (*closer)(hid_t){nullptr};

    ScopedH5() = default;
    ScopedH5(const hid_t handle_id, herr_t (*close_fn)(hid_t)) : id(handle_id), closer(close_fn) {}

    ScopedH5(const ScopedH5&) = delete;
    ScopedH5& operator=(const ScopedH5&) = delete;

    ScopedH5(ScopedH5&& other) noexcept : id(other.id), closer(other.closer) {
        other.id = -1;
        other.closer = nullptr;
    }

    ScopedH5& operator=(ScopedH5&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        id = other.id;
        closer = other.closer;
        other.id = -1;
        other.closer = nullptr;
        return *this;
    }

    ~ScopedH5() { reset(); }

    void reset() noexcept {
        if (id >= 0 && closer != nullptr) {
            static_cast<void>(closer(id));
        }
        id = -1;
        closer = nullptr;
    }
};

void check_status(herr_t status, const char* message);

[[nodiscard]] hid_t check_id(hid_t id, const char* message);

[[nodiscard]] bool link_exists(hid_t parent_id, const std::string& name);

[[nodiscard]] ScopedH5 open_rw_or_create_file(const std::string& path);

[[nodiscard]] ScopedH5 open_ro_file(const std::string& path);

[[nodiscard]] std::vector<std::string> split_path(const std::string& path);

[[nodiscard]] ScopedH5 open_or_create_group(hid_t parent_id, const std::string& name);

[[nodiscard]] ScopedH5 open_group(hid_t parent_id, const std::string& name);

[[nodiscard]] ScopedH5 open_or_create_group_path(hid_t root_id, const std::string& path);

[[nodiscard]] ScopedH5 open_group_path(hid_t root_id, const std::string& path);

void delete_dataset_if_exists(hid_t group_id, const std::string& dataset_name);

template <typename Record>
inline void write_compound_dataset(const hid_t group_id, const std::string& dataset_name,
                                   const hid_t compound_type, const std::vector<Record>& records) {
    delete_dataset_if_exists(group_id, dataset_name);

    hsize_t dims[1]{static_cast<hsize_t>(records.size())};
    auto space = ScopedH5{
        check_id(H5Screate_simple(1, dims, nullptr), "Failed to create HDF5 dataspace"), H5Sclose};

    auto dataset = ScopedH5{check_id(H5Dcreate2(group_id, dataset_name.c_str(), compound_type,
                                                space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                     "Failed to create HDF5 dataset"),
                            H5Dclose};

    if (!records.empty()) {
        check_status(
            H5Dwrite(dataset.id, compound_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, records.data()),
            "Failed to write HDF5 dataset");
    }
}

template <typename Record>
inline std::vector<Record> read_compound_dataset(const hid_t group_id,
                                                 const std::string& dataset_name,
                                                 const hid_t compound_type) {
    auto dataset = ScopedH5{check_id(H5Dopen2(group_id, dataset_name.c_str(), H5P_DEFAULT),
                                     "Failed to open HDF5 dataset"),
                            H5Dclose};
    auto space =
        ScopedH5{check_id(H5Dget_space(dataset.id), "Failed to read HDF5 dataspace"), H5Sclose};

    hsize_t dims[1]{0U};
    check_status(H5Sget_simple_extent_dims(space.id, dims, nullptr),
                 "Failed to read HDF5 dataset dimensions");

    std::vector<Record> records(static_cast<std::size_t>(dims[0]));
    if (!records.empty()) {
        check_status(
            H5Dread(dataset.id, compound_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, records.data()),
            "Failed to read HDF5 dataset");
    }

    return records;
}

struct SpikeEventRecord {
    std::uint32_t source{};
    std::uint32_t target{};
    float arrival{};
    float value{};
};

struct SynapseRecord {
    std::uint32_t pre_id{};
    std::uint32_t post_id{};
    float weight{};
    float delay{};
    std::int8_t sign{};
};

struct NeuronRecord {
    std::uint32_t id{};
    std::uint16_t x{};
    std::uint16_t y{};
    std::uint16_t z{};
    std::uint8_t type{};

    float v_rest{};
    float v_reset{};
    float tau_m{};
    float t_ref{};
    float theta_base{};

    float potential{};
    float threshold{};
    float last_update_time{};
    float last_spike_time{};
    float average_rate{};
    std::uint8_t in_refractory{};
};

struct MetricRecord {
    char name[kMetricNameBytes]{};
    double value{};
};

[[nodiscard]] hid_t make_spike_event_type();

[[nodiscard]] hid_t make_synapse_type();

[[nodiscard]] hid_t make_neuron_type();

[[nodiscard]] hid_t make_metric_type();

[[nodiscard]] SpikeEventRecord to_record(const senna::core::domain::SpikeEvent& event);

[[nodiscard]] senna::core::domain::SpikeEvent from_record(const SpikeEventRecord& record);

[[nodiscard]] SynapseRecord to_record(const senna::core::domain::Synapse& synapse);

[[nodiscard]] senna::core::domain::Synapse from_record(const SynapseRecord& record);

[[nodiscard]] NeuronRecord to_record(const senna::core::domain::NeuronSnapshot& neuron);

[[nodiscard]] senna::core::domain::NeuronSnapshot from_record(const NeuronRecord& record);

[[nodiscard]] MetricRecord to_record(const MetricPoint& metric);

[[nodiscard]] MetricPoint from_record(const MetricRecord& record);

}  // namespace detail

class HDF5Writer final {
   public:
    explicit HDF5Writer(std::string file_path);

    void write_spike_trace(std::size_t epoch,
                           const std::vector<senna::core::domain::SpikeEvent>& trace) const;

    void write_snapshot(std::size_t epoch, const std::vector<senna::core::domain::Neuron>& neurons,
                        const senna::core::domain::SynapseStore& synapses) const;

    void write_snapshot(std::size_t epoch,
                        const std::vector<senna::core::domain::NeuronSnapshot>& neurons,
                        const std::vector<senna::core::domain::Synapse>& synapses) const;

    void write_metrics(std::size_t epoch, const std::vector<MetricPoint>& metrics) const;

    void write_metrics(std::size_t epoch,
                       const std::unordered_map<std::string, double>& metrics) const;

    [[nodiscard]] std::vector<senna::core::domain::SpikeEvent> read_spike_trace(
        std::size_t epoch) const;

    [[nodiscard]] SnapshotData read_snapshot(std::size_t epoch) const;

    [[nodiscard]] std::vector<MetricPoint> read_metrics(std::size_t epoch) const;

    [[nodiscard]] const std::string& file_path() const noexcept { return file_path_; }

   private:
    [[nodiscard]] static std::string epoch_group_path(std::size_t epoch);

    std::string file_path_{};
};

}  // namespace senna::core::persistence

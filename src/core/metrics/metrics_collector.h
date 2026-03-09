#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/types.h"

namespace senna::core::metrics {

struct MetricsSnapshot {
    std::size_t ticks_total{0U};
    std::size_t spikes_total{0U};

    double spikes_per_tick{0.0};
    double mean_spikes_per_tick{0.0};

    double active_neurons_ratio{0.0};
    double max_active_neurons_ratio{0.0};
    double mean_active_neurons_ratio{0.0};

    double e_rate_hz{0.0};
    double i_rate_hz{0.0};
    double ei_balance{0.0};

    double train_accuracy{0.0};
    double test_accuracy{0.0};

    std::size_t synapse_count{0U};
    std::size_t stdp_updates_total{0U};
    std::size_t pruned_total{0U};
    std::size_t sprouted_total{0U};

    double tick_duration_seconds{0.0};
};

class MetricsCollector final {
   public:
    explicit MetricsCollector(const std::vector<senna::core::domain::Neuron>& neurons,
                              const std::size_t synapse_count = 0U) noexcept;

    void on_spike(const senna::core::domain::SpikeEvent& spike);

    void on_tick(senna::core::domain::Time t_start, senna::core::domain::Time t_end);

    void set_train_accuracy(double value) noexcept;

    void set_test_accuracy(double value) noexcept;

    void set_synapse_count(const std::size_t value) noexcept { snapshot_.synapse_count = value; }

    void record_stdp_updates(std::size_t updates) noexcept;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void record_structural_changes(std::size_t pruned, std::size_t sprouted) noexcept;

    void reset() noexcept;

    [[nodiscard]] const MetricsSnapshot& snapshot() const noexcept { return snapshot_; }

    [[nodiscard]] std::unordered_map<std::string, double> as_metric_map() const;

   private:
    [[nodiscard]] static double clamp_0_1(double value) noexcept;

    [[nodiscard]] std::pair<double, double> average_rates_by_type() const noexcept;

    void reset_tick_accumulators() noexcept;

    const std::vector<senna::core::domain::Neuron>& neurons_;

    std::size_t spikes_this_tick_{0U};
    std::size_t active_count_this_tick_{0U};
    std::vector<std::uint8_t> active_neurons_bitmap_{};
    std::vector<senna::core::domain::NeuronId> active_neurons_dirty_{};

    double cumulative_spikes_per_tick_{0.0};
    double cumulative_active_ratio_{0.0};

    MetricsSnapshot snapshot_{};
};

}  // namespace senna::core::metrics

#include "core/metrics/metrics_collector.h"

namespace senna::core::metrics {

MetricsCollector::MetricsCollector(const std::vector<senna::core::domain::Neuron>& neurons,
                                   const std::size_t synapse_count) noexcept
    : neurons_(neurons), active_neurons_bitmap_(neurons.size(), 0U) {
    snapshot_.synapse_count = synapse_count;
}

void MetricsCollector::on_spike(const senna::core::domain::SpikeEvent& spike) {
    ++spikes_this_tick_;

    const auto index = static_cast<std::size_t>(spike.source);
    if (index >= active_neurons_bitmap_.size() || active_neurons_bitmap_[index] != 0U) {
        return;
    }

    active_neurons_bitmap_[index] = 1U;
    active_neurons_dirty_.push_back(spike.source);
    ++active_count_this_tick_;
}

void MetricsCollector::on_tick(const senna::core::domain::Time t_start,
                               const senna::core::domain::Time t_end) {
    snapshot_.spikes_per_tick = static_cast<double>(spikes_this_tick_);

    const auto total_neurons = neurons_.size();
    snapshot_.active_neurons_ratio =
        total_neurons == 0U
            ? 0.0
            : static_cast<double>(active_count_this_tick_) / static_cast<double>(total_neurons);
    snapshot_.max_active_neurons_ratio =
        std::max(snapshot_.max_active_neurons_ratio, snapshot_.active_neurons_ratio);

    const auto [e_rate_hz, i_rate_hz] = average_rates_by_type();
    snapshot_.e_rate_hz = e_rate_hz;
    snapshot_.i_rate_hz = i_rate_hz;
    snapshot_.ei_balance = i_rate_hz > 0.0 ? e_rate_hz / i_rate_hz : e_rate_hz;

    const auto tick_ms = std::max(0.0F, t_end - t_start);
    snapshot_.tick_duration_seconds = static_cast<double>(tick_ms) / 1000.0;

    ++snapshot_.ticks_total;
    snapshot_.spikes_total += spikes_this_tick_;

    cumulative_spikes_per_tick_ += snapshot_.spikes_per_tick;
    cumulative_active_ratio_ += snapshot_.active_neurons_ratio;

    const auto tick_count = static_cast<double>(snapshot_.ticks_total);
    snapshot_.mean_spikes_per_tick = cumulative_spikes_per_tick_ / tick_count;
    snapshot_.mean_active_neurons_ratio = cumulative_active_ratio_ / tick_count;

    reset_tick_accumulators();
}

void MetricsCollector::set_train_accuracy(const double value) noexcept {
    snapshot_.train_accuracy = clamp_0_1(value);
}

void MetricsCollector::set_test_accuracy(const double value) noexcept {
    snapshot_.test_accuracy = clamp_0_1(value);
}

void MetricsCollector::record_stdp_updates(const std::size_t updates) noexcept {
    snapshot_.stdp_updates_total += updates;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void MetricsCollector::record_structural_changes(const std::size_t pruned,
                                                 const std::size_t sprouted) noexcept {
    snapshot_.pruned_total += pruned;
    snapshot_.sprouted_total += sprouted;
}

void MetricsCollector::reset() noexcept {
    reset_tick_accumulators();
    cumulative_spikes_per_tick_ = 0.0;
    cumulative_active_ratio_ = 0.0;
    snapshot_ = MetricsSnapshot{};
}

std::unordered_map<std::string, double> MetricsCollector::as_metric_map() const {
    return {
        {"senna_active_neurons_ratio", snapshot_.active_neurons_ratio},
        {"senna_max_active_neurons_ratio", snapshot_.max_active_neurons_ratio},
        {"senna_spikes_per_tick", snapshot_.spikes_per_tick},
        {"senna_e_rate_hz", snapshot_.e_rate_hz},
        {"senna_i_rate_hz", snapshot_.i_rate_hz},
        {"senna_ei_balance", snapshot_.ei_balance},
        {"senna_train_accuracy", snapshot_.train_accuracy},
        {"senna_test_accuracy", snapshot_.test_accuracy},
        {"senna_synapse_count", static_cast<double>(snapshot_.synapse_count)},
        {"senna_pruned_total", static_cast<double>(snapshot_.pruned_total)},
        {"senna_sprouted_total", static_cast<double>(snapshot_.sprouted_total)},
        {"senna_stdp_updates_total", static_cast<double>(snapshot_.stdp_updates_total)},
        {"senna_tick_duration_seconds", snapshot_.tick_duration_seconds},
    };
}

double MetricsCollector::clamp_0_1(const double value) noexcept {
    return std::clamp(value, 0.0, 1.0);
}

std::pair<double, double> MetricsCollector::average_rates_by_type() const noexcept {
    double e_rate_sum = 0.0;
    double i_rate_sum = 0.0;
    std::size_t e_count = 0U;
    std::size_t i_count = 0U;

    for (const auto& neuron : neurons_) {
        if (neuron.type() == senna::core::domain::NeuronType::Excitatory) {
            e_rate_sum += static_cast<double>(neuron.average_rate());
            ++e_count;
        } else {
            i_rate_sum += static_cast<double>(neuron.average_rate());
            ++i_count;
        }
    }

    const auto e_avg = e_count == 0U ? 0.0 : e_rate_sum / static_cast<double>(e_count);
    const auto i_avg = i_count == 0U ? 0.0 : i_rate_sum / static_cast<double>(i_count);
    return {e_avg, i_avg};
}

void MetricsCollector::reset_tick_accumulators() noexcept {
    spikes_this_tick_ = 0U;
    active_count_this_tick_ = 0U;
    for (const auto neuron_id : active_neurons_dirty_) {
        const auto index = static_cast<std::size_t>(neuron_id);
        if (index < active_neurons_bitmap_.size()) {
            active_neurons_bitmap_[index] = 0U;
        }
    }
    active_neurons_dirty_.clear();
}

}  // namespace senna::core::metrics

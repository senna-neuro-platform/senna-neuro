#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/types.h"
#include "core/metrics/metrics_collector.h"

namespace {

std::vector<senna::core::domain::Neuron> make_neurons() {
    using senna::core::domain::Coord3D;
    using senna::core::domain::Neuron;
    using senna::core::domain::NeuronConfig;
    using senna::core::domain::NeuronType;

    std::vector<Neuron> neurons{};
    neurons.emplace_back(0U, Coord3D{0U, 0U, 0U}, NeuronType::Excitatory, NeuronConfig{});
    neurons.emplace_back(1U, Coord3D{1U, 0U, 0U}, NeuronType::Inhibitory, NeuronConfig{});
    neurons.emplace_back(2U, Coord3D{0U, 1U, 0U}, NeuronType::Excitatory, NeuronConfig{});
    neurons.emplace_back(3U, Coord3D{1U, 1U, 0U}, NeuronType::Inhibitory, NeuronConfig{});
    return neurons;
}

double metric_or_throw(const std::unordered_map<std::string, double>& metrics,
                       const std::string& key) {
    const auto it = metrics.find(key);
    if (it == metrics.end()) {
        throw std::runtime_error("metric key not found: " + key);
    }
    return it->second;
}

}  // namespace

TEST(MetricsCollectorTest, AggregatesExpectedValuesForKnownTickSequence) {
    auto neurons = make_neurons();
    senna::core::metrics::MetricsCollector collector{neurons, 1200U};

    collector.set_train_accuracy(0.91);
    collector.set_test_accuracy(0.88);

    for (std::size_t tick = 0U; tick < 100U; ++tick) {
        const auto t = static_cast<senna::core::domain::Time>(tick) * 0.5F;
        collector.on_spike({0U, 0U, t, 1.0F});
        collector.on_spike({1U, 1U, t, -1.0F});
        collector.on_spike({0U, 0U, t, 1.0F});

        collector.record_stdp_updates(2U);
        if (tick % 20U == 0U) {
            collector.record_structural_changes(1U, 3U);
        }

        collector.on_tick(t, t + 0.5F);
    }

    const auto& snapshot = collector.snapshot();

    EXPECT_EQ(snapshot.ticks_total, 100U);
    EXPECT_EQ(snapshot.spikes_total, 300U);

    EXPECT_DOUBLE_EQ(snapshot.spikes_per_tick, 3.0);
    EXPECT_DOUBLE_EQ(snapshot.mean_spikes_per_tick, 3.0);

    EXPECT_DOUBLE_EQ(snapshot.active_neurons_ratio, 0.5);
    EXPECT_DOUBLE_EQ(snapshot.max_active_neurons_ratio, 0.5);
    EXPECT_DOUBLE_EQ(snapshot.mean_active_neurons_ratio, 0.5);

    EXPECT_NEAR(snapshot.e_rate_hz, 2000.0, 1e-9);
    EXPECT_NEAR(snapshot.i_rate_hz, 1000.0, 1e-9);
    EXPECT_NEAR(snapshot.ei_balance, 2.0, 1e-9);

    EXPECT_EQ(snapshot.synapse_count, 1200U);
    EXPECT_EQ(snapshot.stdp_updates_total, 200U);
    EXPECT_EQ(snapshot.pruned_total, 5U);
    EXPECT_EQ(snapshot.sprouted_total, 15U);

    EXPECT_DOUBLE_EQ(snapshot.train_accuracy, 0.91);
    EXPECT_DOUBLE_EQ(snapshot.test_accuracy, 0.88);
    EXPECT_DOUBLE_EQ(snapshot.tick_duration_seconds, 0.0005);
}

TEST(MetricsCollectorTest, ExportsPrometheusMetricMapWithExpectedKeys) {
    auto neurons = make_neurons();
    senna::core::metrics::MetricsCollector collector{neurons, 42U};

    collector.set_train_accuracy(1.2);
    collector.set_test_accuracy(-0.2);
    collector.record_stdp_updates(5U);
    collector.record_structural_changes(2U, 1U);

    collector.on_spike({0U, 0U, 0.0F, 1.0F});
    collector.on_tick(0.0F, 0.5F);

    const auto metrics = collector.as_metric_map();

    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_active_neurons_ratio"), 0.25);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_max_active_neurons_ratio"), 0.25);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_spikes_per_tick"), 1.0);
    EXPECT_NEAR(metric_or_throw(metrics, "senna_e_rate_hz"), 1000.0, 1e-9);
    EXPECT_NEAR(metric_or_throw(metrics, "senna_i_rate_hz"), 0.0, 1e-9);
    EXPECT_NEAR(metric_or_throw(metrics, "senna_ei_balance"), 1000.0, 1e-9);

    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_train_accuracy"), 1.0);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_test_accuracy"), 0.0);

    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_synapse_count"), 42.0);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_pruned_total"), 2.0);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_sprouted_total"), 1.0);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_stdp_updates_total"), 5.0);
    EXPECT_DOUBLE_EQ(metric_or_throw(metrics, "senna_tick_duration_seconds"), 0.0005);
}

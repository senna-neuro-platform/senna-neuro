/**
 * @file metrics_collector.hpp
 * @brief Lock-free event collector for observability pipeline (Шаг 11.1).
 *
 * Producer-facing API only: core subsystems enqueue lightweight metric events
 * without blocking the spike loop. A single consumer thread (observability
 * worker) drains events later for aggregation / export (Prometheus, logs,
 * etc.).
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace senna::observability {

class MetricsCollector {
 public:
  enum class EventType {
    kTickSummary,
    kWeightUpdate,
    kPrune,
    kSprout,
    kRateSample,
    kSynapseCount,
    kPhaseSample,
    kAccuracy,
    kMemory,
    kVirtualTime,
  };

  struct TickSummary {
    uint64_t tick_id{0};
    uint32_t active_neurons{0};
    uint32_t spikes{0};
    double tick_duration_ms{0.0};
  };

  struct WeightUpdate {
    float delta{0.0F};
    float new_weight{0.0F};
  };

  struct PruneEvent {
    uint32_t count{0};
  };

  struct SproutEvent {
    uint32_t count{0};
  };

  struct RateSample {
    double excitatory_hz{0.0};
    double inhibitory_hz{0.0};
  };

  struct SynapseCount {
    uint64_t count{0};
  };

  struct PhaseSample {
    double phase{0.0};           // 0=awake, 1=sleep
    double sleep_pressure{0.0};  // arbitrary unit
  };

  struct AccuracySample {
    double train{0.0};
    double test{0.0};
  };

  struct MemorySample {
    uint64_t bytes{0};
  };

  struct VirtualTimeSample {
    uint64_t ms{0};
  };

  struct Event {
    Event()
        : type(EventType::kTickSummary),
          timestamp_ns(0),
          payload(TickSummary{}) {}
    Event(EventType t, uint64_t ts,
          std::variant<TickSummary, WeightUpdate, PruneEvent, SproutEvent,
                       RateSample, SynapseCount, PhaseSample, AccuracySample,
                       MemorySample, VirtualTimeSample>
              p)
        : type(t), timestamp_ns(ts), payload(std::move(p)) {}

    EventType type{EventType::kTickSummary};
    uint64_t timestamp_ns{0};
    std::variant<TickSummary, WeightUpdate, PruneEvent, SproutEvent, RateSample,
                 SynapseCount, PhaseSample, AccuracySample, MemorySample,
                 VirtualTimeSample>
        payload;
  };

  explicit MetricsCollector(std::size_t capacity = 8192);

  // --- Producer API (non-blocking, drop-on-full) ---
  bool RecordTickSummary(uint64_t tick_id, uint32_t active_neurons,
                         uint32_t spikes, double tick_duration_ms);
  bool RecordWeightUpdate(float delta, float new_weight);
  bool RecordPrune(uint32_t count = 1);
  bool RecordSprout(uint32_t count = 1);
  bool RecordRates(double excitatory_hz, double inhibitory_hz);
  bool RecordSynapseCount(uint64_t count);
  bool RecordPhase(double phase, double sleep_pressure);
  bool RecordAccuracy(double train, double test);
  bool RecordMemoryBytes(uint64_t bytes);
  bool RecordVirtualTimeMs(uint64_t ms);

  // --- Consumer API (single consumer thread) ---
  std::optional<Event> Pop();
  std::size_t Pending() const;
  std::size_t Dropped() const {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  bool Push(const Event& event);
  static uint64_t NowNs();

  const std::size_t capacity_;
  std::vector<Event> buffer_;
  std::atomic<uint64_t> write_idx_{0};
  std::atomic<uint64_t> read_idx_{0};
  std::atomic<std::size_t> dropped_{0};
};

struct AggregatedMetrics {
  uint64_t last_tick_id{0};
  uint32_t last_active_neurons{0};
  uint32_t last_spikes{0};
  double last_tick_duration_ms{0.0};
  double active_ratio{0.0};
  double spikes_per_tick{0.0};
  double excitatory_rate_hz{0.0};
  double inhibitory_rate_hz{0.0};
  double ei_balance{0.0};
  uint64_t pruned_total{0};
  uint64_t sprouted_total{0};
  uint64_t synapse_count{0};
  float last_weight{0.0F};
  float last_weight_delta{0.0F};
  double train_accuracy{0.0};
  double test_accuracy{0.0};
  double ticks_per_second{0.0};
  uint64_t memory_bytes{0};
  double phase{0.0};
  double sleep_pressure{0.0};
  uint64_t virtual_time_ms{0};

  std::vector<double> tick_duration_buckets;   // edges (seconds)
  std::vector<uint64_t> tick_duration_counts;  // same size +1 for +Inf
  uint64_t tick_duration_count{0};
  double tick_duration_sum{0.0};  // seconds
};

// Background worker that aggregates events from MetricsCollector.
class ObservabilityThread {
 public:
  ObservabilityThread(
      MetricsCollector& collector, uint32_t total_neurons,
      std::chrono::milliseconds idle_sleep = std::chrono::milliseconds(2),
      std::vector<double> tick_duration_buckets = {});
  ~ObservabilityThread();

  void Start();
  void Stop();

  AggregatedMetrics Snapshot() const;
  // Helper to format snapshot into Prometheus text exposition.
  static std::string RenderPrometheus(const AggregatedMetrics& agg);

 private:
  void Run();

  MetricsCollector& collector_;
  const uint32_t total_neurons_;
  const std::chrono::milliseconds idle_sleep_;

  std::atomic<bool> running_{false};
  std::thread worker_;

  mutable std::mutex mtx_;
  AggregatedMetrics agg_{};
};

// Convenience helper to plug into PrometheusExporter.
inline auto MakePrometheusRender(ObservabilityThread& obs) {
  return [&obs]() {
    return ObservabilityThread::RenderPrometheus(obs.Snapshot());
  };
}

}  // namespace senna::observability

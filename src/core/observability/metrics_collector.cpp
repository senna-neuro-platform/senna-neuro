/**
 * @file metrics_collector.cpp
 * @brief Lock-free, drop-on-full ring buffer for observability events.
 */

#include "core/observability/metrics_collector.hpp"

#include <string>
#include <utility>

namespace senna::observability {
namespace {
constexpr std::size_t kDefaultCapacity = 8192;
}  // namespace

MetricsCollector::MetricsCollector(std::size_t capacity)
    : capacity_(capacity == 0 ? kDefaultCapacity : capacity),
      buffer_(capacity_) {}

bool MetricsCollector::RecordTickSummary(uint64_t tick_id,
                                         uint32_t active_neurons,
                                         uint32_t spikes,
                                         double tick_duration_ms) {
  Event ev{EventType::kTickSummary, NowNs(),
           TickSummary{tick_id, active_neurons, spikes, tick_duration_ms}};
  return Push(ev);
}

bool MetricsCollector::RecordWeightUpdate(float delta, float new_weight) {
  Event ev{EventType::kWeightUpdate, NowNs(), WeightUpdate{delta, new_weight}};
  return Push(ev);
}

bool MetricsCollector::RecordPrune(uint32_t count) {
  Event ev{EventType::kPrune, NowNs(), PruneEvent{count}};
  return Push(ev);
}

bool MetricsCollector::RecordSprout(uint32_t count) {
  Event ev{EventType::kSprout, NowNs(), SproutEvent{count}};
  return Push(ev);
}

bool MetricsCollector::RecordRates(double excitatory_hz, double inhibitory_hz) {
  Event ev{EventType::kRateSample, NowNs(),
           RateSample{excitatory_hz, inhibitory_hz}};
  return Push(ev);
}

bool MetricsCollector::RecordSynapseCount(uint64_t count) {
  Event ev{EventType::kSynapseCount, NowNs(), SynapseCount{count}};
  return Push(ev);
}

bool MetricsCollector::RecordPhase(double phase, double sleep_pressure) {
  Event ev{EventType::kPhaseSample, NowNs(),
           PhaseSample{phase, sleep_pressure}};
  return Push(ev);
}

bool MetricsCollector::RecordAccuracy(double train, double test) {
  Event ev{EventType::kAccuracy, NowNs(), AccuracySample{train, test}};
  return Push(ev);
}

bool MetricsCollector::RecordMemoryBytes(uint64_t bytes) {
  Event ev{EventType::kMemory, NowNs(), MemorySample{bytes}};
  return Push(ev);
}

bool MetricsCollector::RecordVirtualTimeMs(uint64_t ms) {
  Event ev{EventType::kVirtualTime, NowNs(), VirtualTimeSample{ms}};
  return Push(ev);
}

std::optional<MetricsCollector::Event> MetricsCollector::Pop() {
  const uint64_t r = read_idx_.load(std::memory_order_relaxed);
  const uint64_t w = write_idx_.load(std::memory_order_acquire);
  if (r >= w) {
    return std::nullopt;
  }

  auto idx = static_cast<std::size_t>(r % capacity_);
  Event ev = buffer_[idx];
  read_idx_.store(r + 1, std::memory_order_release);
  return ev;
}

std::size_t MetricsCollector::Pending() const {
  const uint64_t w = write_idx_.load(std::memory_order_acquire);
  const uint64_t r = read_idx_.load(std::memory_order_relaxed);
  return static_cast<std::size_t>(w - r);
}

bool MetricsCollector::Push(const Event& event) {
  const uint64_t w = write_idx_.load(std::memory_order_relaxed);
  const uint64_t r = read_idx_.load(std::memory_order_acquire);

  if (static_cast<std::size_t>(w - r) >= capacity_) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;  // drop silently when the ring is full
  }

  auto idx = static_cast<std::size_t>(w % capacity_);
  buffer_[idx] = event;
  write_idx_.store(w + 1, std::memory_order_release);
  return true;
}

uint64_t MetricsCollector::NowNs() {
  using clock = std::chrono::steady_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          clock::now().time_since_epoch())
          .count());
}

// -------- ObservabilityThread ----------

ObservabilityThread::ObservabilityThread(
    MetricsCollector& collector, uint32_t total_neurons,
    std::chrono::milliseconds idle_sleep,
    std::vector<double> tick_duration_buckets)
    : collector_(collector),
      total_neurons_(total_neurons),
      idle_sleep_(idle_sleep) {
  if (tick_duration_buckets.empty()) {
    agg_.tick_duration_buckets =
        std::vector<double>{0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1};
  } else {
    agg_.tick_duration_buckets = std::move(tick_duration_buckets);
  }
  agg_.tick_duration_counts.assign(agg_.tick_duration_buckets.size() + 1, 0);
}

ObservabilityThread::~ObservabilityThread() { Stop(); }

void ObservabilityThread::Start() {
  if (running_.exchange(true)) {
    return;
  }
  worker_ = std::thread(&ObservabilityThread::Run, this);
}

void ObservabilityThread::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

AggregatedMetrics ObservabilityThread::Snapshot() const {
  std::lock_guard<std::mutex> lk(mtx_);
  return agg_;
}

void ObservabilityThread::Run() {
  while (running_.load(std::memory_order_relaxed)) {
    bool processed = false;
    while (auto ev_opt = collector_.Pop()) {
      processed = true;
      const MetricsCollector::Event& ev = *ev_opt;
      std::lock_guard<std::mutex> lk(mtx_);
      switch (ev.type) {
        case MetricsCollector::EventType::kTickSummary: {
          const auto& t = std::get<MetricsCollector::TickSummary>(ev.payload);
          agg_.last_tick_id = t.tick_id;
          agg_.last_active_neurons = t.active_neurons;
          agg_.last_spikes = t.spikes;
          agg_.last_tick_duration_ms = t.tick_duration_ms;
          agg_.active_ratio = total_neurons_ > 0
                                  ? static_cast<double>(t.active_neurons) /
                                        static_cast<double>(total_neurons_)
                                  : 0.0;
          agg_.spikes_per_tick = t.spikes;
          const double dur_s = t.tick_duration_ms / 1000.0;
          agg_.tick_duration_sum += dur_s;
          agg_.tick_duration_count += 1;
          bool placed = false;
          for (std::size_t i = 0; i < agg_.tick_duration_buckets.size(); ++i) {
            if (dur_s <= agg_.tick_duration_buckets[i]) {
              agg_.tick_duration_counts[i] += 1;
              placed = true;
              break;
            }
          }
          if (!placed) {
            agg_.tick_duration_counts.back() += 1;  // +Inf bucket
          }
          if (dur_s > 0.0) {
            agg_.ticks_per_second = 1.0 / dur_s;
          }
          break;
        }
        case MetricsCollector::EventType::kWeightUpdate: {
          const auto& w = std::get<MetricsCollector::WeightUpdate>(ev.payload);
          agg_.last_weight_delta = w.delta;
          agg_.last_weight = w.new_weight;
          break;
        }
        case MetricsCollector::EventType::kPrune: {
          const auto& p = std::get<MetricsCollector::PruneEvent>(ev.payload);
          agg_.pruned_total += p.count;
          break;
        }
        case MetricsCollector::EventType::kSprout: {
          const auto& s = std::get<MetricsCollector::SproutEvent>(ev.payload);
          agg_.sprouted_total += s.count;
          break;
        }
        case MetricsCollector::EventType::kRateSample: {
          const auto& r = std::get<MetricsCollector::RateSample>(ev.payload);
          agg_.excitatory_rate_hz = r.excitatory_hz;
          agg_.inhibitory_rate_hz = r.inhibitory_hz;
          agg_.ei_balance =
              r.inhibitory_hz != 0.0 ? r.excitatory_hz / r.inhibitory_hz : 0.0;
          break;
        }
        case MetricsCollector::EventType::kSynapseCount: {
          const auto& sc = std::get<MetricsCollector::SynapseCount>(ev.payload);
          agg_.synapse_count = sc.count;
          break;
        }
        case MetricsCollector::EventType::kPhaseSample: {
          const auto& ph = std::get<MetricsCollector::PhaseSample>(ev.payload);
          agg_.phase = ph.phase;
          agg_.sleep_pressure = ph.sleep_pressure;
          break;
        }
        case MetricsCollector::EventType::kAccuracy: {
          const auto& acc =
              std::get<MetricsCollector::AccuracySample>(ev.payload);
          agg_.train_accuracy = acc.train;
          agg_.test_accuracy = acc.test;
          break;
        }
        case MetricsCollector::EventType::kMemory: {
          const auto& mem =
              std::get<MetricsCollector::MemorySample>(ev.payload);
          agg_.memory_bytes = mem.bytes;
          break;
        }
        case MetricsCollector::EventType::kVirtualTime: {
          const auto& vt =
              std::get<MetricsCollector::VirtualTimeSample>(ev.payload);
          agg_.virtual_time_ms = vt.ms;
          break;
        }
      }
    }

    if (!processed) {
      std::this_thread::sleep_for(idle_sleep_);
    }
  }
}

std::string ObservabilityThread::RenderPrometheus(
    const AggregatedMetrics& agg) {
  std::string out;
  auto append_gauge = [&out](const std::string& name, const std::string& help,
                             double value) {
    out += "# HELP " + name + " " + help + "\n";
    out += "# TYPE " + name + " gauge\n";
    out += name + " " + std::to_string(value) + "\n";
  };

  append_gauge("senna_active_neurons_ratio", "Ratio of active neurons per tick",
               agg.active_ratio);
  append_gauge("senna_spikes_per_tick", "Spikes per tick", agg.spikes_per_tick);
  append_gauge("senna_excitatory_rate_avg",
               "Average excitatory firing rate (Hz)", agg.excitatory_rate_hz);
  append_gauge("senna_inhibitory_rate_avg",
               "Average inhibitory firing rate (Hz)", agg.inhibitory_rate_hz);
  append_gauge("senna_ei_balance", "E/I balance (E_rate / I_rate)",
               agg.ei_balance);
  append_gauge("senna_train_accuracy", "Training accuracy", agg.train_accuracy);
  append_gauge("senna_test_accuracy", "Test accuracy", agg.test_accuracy);
  append_gauge("senna_synapse_count", "Current synapse count",
               static_cast<double>(agg.synapse_count));
  append_gauge("senna_pruned_total", "Total pruned synapses",
               static_cast<double>(agg.pruned_total));
  append_gauge("senna_sprouted_total", "Total sprouted synapses",
               static_cast<double>(agg.sprouted_total));
  append_gauge("senna_ticks_per_second", "Ticks per second",
               agg.ticks_per_second);
  append_gauge("senna_memory_bytes", "Memory usage in bytes",
               static_cast<double>(agg.memory_bytes));
  append_gauge("senna_current_phase", "Phase: 0=awake,1=sleep", agg.phase);
  append_gauge("senna_sleep_pressure", "Sleep pressure", agg.sleep_pressure);
  append_gauge("senna_virtual_time_ms", "Virtual time (ms)",
               static_cast<double>(agg.virtual_time_ms));

  out += "# HELP senna_tick_duration_seconds Tick duration histogram\n";
  out += "# TYPE senna_tick_duration_seconds histogram\n";
  double cumulative = 0.0;
  for (std::size_t i = 0; i < agg.tick_duration_buckets.size(); ++i) {
    cumulative += static_cast<double>(agg.tick_duration_counts[i]);
    out += "senna_tick_duration_seconds_bucket{le=\"" +
           std::to_string(agg.tick_duration_buckets[i]) + "\"} " +
           std::to_string(cumulative) + "\n";
  }
  cumulative += static_cast<double>(agg.tick_duration_counts.back());
  out += "senna_tick_duration_seconds_bucket{le=\"+Inf\"} " +
         std::to_string(cumulative) + "\n";
  out += "senna_tick_duration_seconds_sum " +
         std::to_string(agg.tick_duration_sum) + "\n";
  out += "senna_tick_duration_seconds_count " +
         std::to_string(agg.tick_duration_count) + "\n";

  return out;
}

}  // namespace senna::observability

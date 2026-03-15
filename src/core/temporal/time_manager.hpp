#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "core/neural/neuron_pool.hpp"
#include "core/observability/metrics_collector.hpp"
#include "core/plasticity/homeostasis.hpp"
#include "core/synaptic/synapse_index.hpp"
#include "core/temporal/event_queue.hpp"

namespace senna::temporal {

// Manages virtual time and the spike-processing loop.
//
// Each tick processes all events in [t, t + dt):
//   1. Drain events from the queue for the current tick
//   2. Deliver events to target neurons (ReceiveInput)
//   3. Collect spikes from neurons that fired
//   4. For each spike, generate new events via outgoing synapses
//   5. Push new events into the queue
//   6. Advance time by dt
class TimeManager {
 public:
  explicit TimeManager(float dt = 0.5F, plasticity::HomeostasisConfig hcfg = {},
                       uint64_t seed = 42);
  ~TimeManager();

  // Run a single tick: drain, deliver, spike, fan-out, advance.
  // Returns the list of neuron IDs that fired this tick.
  std::vector<int32_t> Tick(EventQueue& queue, neural::NeuronPool& pool,
                            const synaptic::SynapseIndex& synapses);

  float time() const { return t_now_; }
  uint64_t tick_count() const { return tick_counter_; }
  float dt() const { return dt_; }
  std::vector<int32_t> LastFiredCopy() const {
    std::scoped_lock lock(last_fired_mutex_);
    return last_fired_;
  }
  std::vector<std::pair<int32_t, float>> LastSpikesCopy() const {
    std::scoped_lock lock(last_fired_mutex_);
    return last_spikes_;
  }
  float last_time() const { return last_t_ms_; }
  struct Snapshot {
    uint64_t tick{0};
    float t_ms{0};
    std::vector<int32_t> fired;
    std::vector<std::pair<int32_t, float>> spikes;
  };
  std::shared_ptr<Snapshot> SnapshotPtr() const {
    std::scoped_lock lock(stream_mutex_);
    return stream_snapshot_;
  }

  void set_time(float t) { t_now_ = t; }
  void set_homeostasis(const plasticity::HomeostasisConfig& cfg) {
    homeostasis_.SetConfig(cfg);
    hcfg_ = cfg;
  }
  void attach_pool(neural::NeuronPool* pool);
  void attach_metrics(observability::MetricsCollector* metrics) {
    metrics_ = metrics;
  }

 private:
  void homeostasis_worker();

  float t_now_ = 0.0F;
  float dt_;
  plasticity::Homeostasis homeostasis_;
  plasticity::HomeostasisConfig hcfg_;
  std::mt19937_64 rng_;

  // Reusable buffers to avoid per-tick allocations.
  std::vector<SpikeEvent> tick_events_;
  std::vector<SpikeEvent> new_events_;

  // Last tick fired ids (for WebSocket streaming).
  std::vector<int32_t> last_fired_;
  std::vector<std::pair<int32_t, float>> last_spikes_;
  float last_t_ms_ = 0.0F;
  mutable std::mutex last_fired_mutex_;
  std::shared_ptr<Snapshot> stream_snapshot_{std::make_shared<Snapshot>()};
  mutable std::mutex stream_mutex_;

  // Homeostasis background thread state.
  neural::NeuronPool* pool_ = nullptr;
  std::thread homeo_thread_;
  std::mutex homeo_mutex_;
  std::condition_variable homeo_cv_;
  bool homeo_stop_ = false;
  int ticks_since_homeo_ = 0;
  struct HomeoTask {
    std::vector<float> theta_snapshot;
    std::vector<float> r_avg_snapshot;
    float global_activity;
  };
  std::deque<HomeoTask> homeo_queue_;

  observability::MetricsCollector* metrics_ = nullptr;
  uint64_t tick_counter_ = 0;
  int excitatory_count_ = 0;
  int inhibitory_count_ = 0;
};

}  // namespace senna::temporal

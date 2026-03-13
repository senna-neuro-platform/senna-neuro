#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "core/neural/neuron_pool.hpp"
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
  float dt() const { return dt_; }

  void set_time(float t) { t_now_ = t; }
  void set_homeostasis(const plasticity::HomeostasisConfig& cfg) {
    homeostasis_.SetConfig(cfg);
    hcfg_ = cfg;
  }
  void attach_pool(neural::NeuronPool* pool);

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
};

}  // namespace senna::temporal

#include "core/temporal/time_manager.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

namespace senna::temporal {

TimeManager::TimeManager(float dt, plasticity::HomeostasisConfig hcfg,
                         uint64_t seed)
    : dt_(dt), homeostasis_(hcfg), hcfg_(hcfg), rng_(seed) {}

std::vector<int32_t> TimeManager::Tick(EventQueue& queue,
                                       neural::NeuronPool& pool,
                                       const synaptic::SynapseIndex& synapses) {
  auto tick_start = std::chrono::steady_clock::now();
  float t_end = t_now_ + dt_;

  // 1. Drain events for this tick [t_now_, t_now_ + dt).
  tick_events_.clear();
  queue.DrainUntil(t_end, tick_events_);

  // Group events by target to allow safe parallel neuron updates.
  std::vector<std::vector<SpikeEvent>> events_by_target(pool.size());
  std::vector<int> active_targets;
  active_targets.reserve(tick_events_.size());
  for (const auto& event : tick_events_) {
    if (events_by_target[event.target_id].empty()) {
      active_targets.push_back(event.target_id);
    }
    events_by_target[event.target_id].push_back(event);
  }

  // Randomize processing order to provide stochastic tie-breaker (WTA).
  std::shuffle(active_targets.begin(), active_targets.end(), rng_);

  // 2. Deliver events to target neurons in parallel and collect fired flags.
  std::vector<uint8_t> fired_flags(pool.size(), 0);
  const unsigned workers = std::max(1U, std::thread::hardware_concurrency());
  const size_t chunk = active_targets.empty()
                           ? 0
                           : (active_targets.size() + workers - 1) / workers;

  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (unsigned w = 0; w < workers; ++w) {
    size_t start = w * chunk;
    if (start >= active_targets.size()) {
      break;
    }
    size_t end = std::min(active_targets.size(), start + chunk);
    threads.emplace_back([&, start, end]() {
      for (size_t idx = start; idx < end; ++idx) {
        int neuron_id = active_targets[idx];
        for (const auto& event : events_by_target[neuron_id]) {
          if (pool.ReceiveInput(neuron_id, event.arrival_time, event.value)) {
            pool.Fire(neuron_id, event.arrival_time);
            fired_flags[neuron_id] = 1;
            break;  // refractory prevents another spike this tick
          }
        }
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }

  std::vector<int32_t> fired;
  fired.reserve(active_targets.size());
  for (int neuron_id : active_targets) {
    if (fired_flags[neuron_id] != 0) {
      fired.push_back(neuron_id);
    }
  }

  // 3. For each spike, generate new events via outgoing synapses.
  new_events_.clear();
  for (int32_t neuron_id : fired) {
    float spike_time = pool.t_spike(neuron_id);
    for (auto sid : synapses.Outgoing(neuron_id)) {
      const auto& syn = synapses.Get(sid);
      new_events_.push_back({
          .target_id = syn.post_id,
          .source_id = syn.pre_id,
          .arrival_time = spike_time + syn.delay,
          .value = syn.Effective(),
      });
    }
  }

  // 4. Push new events into the queue.
  if (!new_events_.empty()) {
    queue.PushBatch(new_events_);
  }

  // 5. Update smoothed firing rates; schedule background homeostasis.
  pool.UpdateAverages(fired, hcfg_.alpha);
  float global_activity = pool.size() > 0 ? static_cast<float>(fired.size()) /
                                                static_cast<float>(pool.size())
                                          : 0.0F;
  if (pool_ != nullptr && hcfg_.interval_ticks > 0) {
    ++ticks_since_homeo_;
    if (ticks_since_homeo_ >= hcfg_.interval_ticks) {
      ticks_since_homeo_ = 0;
      HomeoTask task;
      task.theta_snapshot = pool.ThetaSnapshot();
      task.r_avg_snapshot = pool.RateSnapshot();
      task.global_activity = global_activity;
      {
        std::lock_guard<std::mutex> lk(homeo_mutex_);
        homeo_queue_.push_back(std::move(task));
      }
      homeo_cv_.notify_one();
    }
  }

  // 6. Advance time.
  t_now_ = t_end;

  if (metrics_ != nullptr) {
    const double tick_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - tick_start)
                               .count();

    // Count spikes per type.
    int e_spikes = 0;
    int i_spikes = 0;
    for (int id : fired) {
      if (pool.type(id) == neural::NeuronType::Excitatory) {
        ++e_spikes;
      } else {
        ++i_spikes;
      }
    }
    const double dt_s = static_cast<double>(dt_) / 1000.0;  // dt_ in ms
    double e_rate = (excitatory_count_ > 0 && dt_s > 0.0)
                        ? static_cast<double>(e_spikes) /
                              static_cast<double>(excitatory_count_) / dt_s
                        : 0.0;
    double i_rate = (inhibitory_count_ > 0 && dt_s > 0.0)
                        ? static_cast<double>(i_spikes) /
                              static_cast<double>(inhibitory_count_) / dt_s
                        : 0.0;

    metrics_->RecordTickSummary(tick_counter_,
                                static_cast<uint32_t>(fired.size()),
                                static_cast<uint32_t>(fired.size()), tick_ms);
    metrics_->RecordRates(e_rate, i_rate);
    metrics_->RecordSynapseCount(
        static_cast<uint64_t>(synapses.synapse_count()));
    metrics_->RecordVirtualTimeMs(static_cast<uint64_t>(t_now_));
    // Memory usage (RSS) in bytes.
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      long rss_kb =
          usage.ru_maxrss;  // NOLINT(cppcoreguidelines-pro-type-union-access)
      if (rss_kb > 0) {
        metrics_->RecordMemoryBytes(static_cast<uint64_t>(rss_kb) * 1024ULL);
      }
    }
    ++tick_counter_;
  }

  return fired;
}

void TimeManager::attach_pool(neural::NeuronPool* pool) {
  pool_ = pool;
  if (pool_ != nullptr) {
    excitatory_count_ = 0;
    inhibitory_count_ = 0;
    for (auto t : pool_->type_array()) {
      if (t == neural::NeuronType::Excitatory) {
        ++excitatory_count_;
      } else {
        ++inhibitory_count_;
      }
    }
  }
  if (!homeo_thread_.joinable()) {
    homeo_thread_ = std::thread(&TimeManager::homeostasis_worker, this);
  }
}

void TimeManager::homeostasis_worker() {
  while (true) {
    HomeoTask task;
    {
      std::unique_lock<std::mutex> lk(homeo_mutex_);
      homeo_cv_.wait(lk, [&] { return homeo_stop_ || !homeo_queue_.empty(); });
      if (homeo_stop_) {
        break;
      }
      task = std::move(homeo_queue_.front());
      homeo_queue_.pop_front();
    }
    if (pool_ == nullptr) {
      continue;
    }
    auto theta_new = homeostasis_.ComputeTheta(
        task.theta_snapshot, task.r_avg_snapshot, dt_, task.global_activity);
    pool_->ApplyThetaBuffer(theta_new);
  }
}

TimeManager::~TimeManager() {
  if (homeo_thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(homeo_mutex_);
      homeo_stop_ = true;
    }
    homeo_cv_.notify_all();
    homeo_thread_.join();
  }
}

}  // namespace senna::temporal

#include "core/temporal/time_manager.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace senna::temporal {

TimeManager::TimeManager(float dt, HomeostasisConfig hcfg, uint64_t seed)
    : dt_(dt), hcfg_(hcfg), rng_(seed) {}

std::vector<int32_t> TimeManager::Tick(EventQueue& queue,
                                       neural::NeuronPool& pool,
                                       const synaptic::SynapseIndex& synapses) {
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
  const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
  const size_t chunk = active_targets.empty()
                           ? 0
                           : (active_targets.size() + workers - 1) / workers;

  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (unsigned w = 0; w < workers; ++w) {
    size_t start = w * chunk;
    if (start >= active_targets.size()) break;
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
  for (auto& t : threads) t.join();

  std::vector<int32_t> fired;
  fired.reserve(active_targets.size());
  for (int neuron_id : active_targets) {
    if (fired_flags[neuron_id]) fired.push_back(neuron_id);
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

  // 5. Homeostasis (optional simple rule).
  float global_activity =
      pool.size() > 0 ? static_cast<float>(fired.size()) / pool.size() : 0.0f;
  pool.ApplyHomeostasis(fired, hcfg_.alpha, hcfg_.target_rate, hcfg_.theta_step,
                        global_activity);

  // 6. Advance time.
  t_now_ = t_end;

  return fired;
}

}  // namespace senna::temporal

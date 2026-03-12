#include "core/network/spike_loop.hpp"

#include <set>

namespace senna::network {

SpikeLoop::SpikeLoop(Network& net) : net_(net) {}

RunStats SpikeLoop::Run(float duration_ms) {
  spike_log_.clear();

  auto& tm = net_.time_manager();
  auto& queue = net_.queue();
  auto& pool = net_.pool();
  const auto& synapses = net_.synapses();

  float t_start = tm.time();
  float t_end = t_start + duration_ms;
  int ticks = 0;

  std::set<int32_t> active_set;

  while (tm.time() < t_end) {
    auto fired = tm.Tick(queue, pool, synapses);

    for (int32_t id : fired) {
      spike_log_.emplace_back(id, pool.t_spike(id));
      active_set.insert(id);
    }
    ++ticks;
  }

  return {
      .total_spikes = static_cast<int>(spike_log_.size()),
      .active_neurons = static_cast<int>(active_set.size()),
      .ticks = ticks,
      .duration_ms = duration_ms,
  };
}

}  // namespace senna::network

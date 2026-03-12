#include "core/network/spike_loop.hpp"

#include <set>
#include <thread>

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

  if (decoder_) {
    decoder_->Reset(tm.time());
    decoder_->SetStartTime(tm.time());
  }

  while (tm.time() < t_end) {
    auto fired = tm.Tick(queue, pool, synapses);

    for (int32_t id : fired) {
      spike_log_.emplace_back(id, pool.t_spike(id));
      active_set.insert(id);
      if (decoder_) decoder_->Observe(id, pool.t_spike(id));
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

RunStats SpikeLoop::RunInThread(float duration_ms) {
  RunStats stats;
  std::thread worker([&]() { stats = Run(duration_ms); });
  worker.join();
  return stats;
}

}  // namespace senna::network

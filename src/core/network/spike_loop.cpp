#include "core/network/spike_loop.hpp"

#include <set>
#include <thread>

#include "core/plasticity/stdp.hpp"

namespace senna::network {

SpikeLoop::SpikeLoop(Network& net) : net_(net) {
  stdp_worker_ = std::make_unique<plasticity::STDPWorker>(
      net_.synapses_ptr_atomic(), net_.pool(), net_.config().synapse_params,
      net_.config().stdp_params, net_.config().seed);
}

RunStats SpikeLoop::Run(float duration_ms) {
  spike_log_.clear();

  auto& tm = net_.time_manager();
  auto& queue = net_.queue();
  auto& pool = net_.pool();
  const auto& cfg = net_.config();

  float t_start = tm.time();
  float t_end = t_start + duration_ms;
  int ticks = 0;

  std::set<int32_t> active_set;

  if (decoder_) {
    decoder_->Reset(tm.time());
    decoder_->SetStartTime(tm.time());
    decoder_->SetSeed(cfg.seed);
    decoder_->SetWindow(cfg.decoder_window_ms);
  }
  if (stdp_worker_) stdp_worker_->Start();

  while (tm.time() < t_end) {
    auto syn_ptr = net_.synapses_ptr();
    auto fired = tm.Tick(queue, pool, *syn_ptr);

    for (int32_t id : fired) {
      spike_log_.emplace_back(id, pool.t_spike(id));
      active_set.insert(id);
      if (decoder_) decoder_->Observe(id, pool.t_spike(id));
      if (stdp_worker_) stdp_worker_->Enqueue(id, pool.t_spike(id));
    }
    if (decoder_) decoder_->Finalize(tm.time());
    ++ticks;

    if (net_.structural_worker() && cfg.structural.interval_ticks > 0 &&
        (ticks % cfg.structural.interval_ticks) == 0) {
      net_.structural_worker()->Trigger();
    }
  }
  if (stdp_worker_) stdp_worker_->Stop();

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

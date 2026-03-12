#include "core/temporal/time_manager.hpp"

namespace senna::temporal {

TimeManager::TimeManager(float dt) : dt_(dt) {}

std::vector<int32_t> TimeManager::Tick(EventQueue& queue,
                                       neural::NeuronPool& pool,
                                       const synaptic::SynapseIndex& synapses) {
  float t_end = t_now_ + dt_;

  // 1. Drain events for this tick [t_now_, t_now_ + dt).
  tick_events_.clear();
  queue.DrainUntil(t_end, tick_events_);

  // 2. Deliver events to target neurons and collect spikes.
  std::vector<int32_t> fired;
  for (const auto& event : tick_events_) {
    bool spiked =
        pool.ReceiveInput(event.target_id, event.arrival_time, event.value);
    if (spiked) {
      pool.Fire(event.target_id, event.arrival_time);
      fired.push_back(event.target_id);
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

  // 5. Advance time.
  t_now_ = t_end;

  return fired;
}

}  // namespace senna::temporal

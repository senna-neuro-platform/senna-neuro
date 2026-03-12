#pragma once

#include <vector>

#include "core/neural/neuron_pool.hpp"
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
  explicit TimeManager(float dt = 0.5f);

  // Run a single tick: drain, deliver, spike, fan-out, advance.
  // Returns the list of neuron IDs that fired this tick.
  std::vector<int32_t> Tick(EventQueue& queue, neural::NeuronPool& pool,
                            const synaptic::SynapseIndex& synapses);

  float time() const { return t_now_; }
  float dt() const { return dt_; }

  void set_time(float t) { t_now_ = t; }

 private:
  float t_now_ = 0.0f;
  float dt_;

  // Reusable buffers to avoid per-tick allocations.
  std::vector<SpikeEvent> tick_events_;
  std::vector<SpikeEvent> new_events_;
};

}  // namespace senna::temporal

#pragma once

#include <atomic>
#include <cstdint>
#include <queue>
#include <vector>

namespace senna::temporal {

// A spike event delivered to a target neuron at a specific time.
struct SpikeEvent {
  int32_t target_id;   // postsynaptic neuron
  int32_t source_id;   // presynaptic neuron (for STDP)
  float arrival_time;  // when the event arrives (ms)
  float value;         // weight * sign (effective contribution)

  // Min-heap comparator: smallest arrival_time has highest priority.
  bool operator>(const SpikeEvent& other) const {
    return arrival_time > other.arrival_time;
  }
};

// Thread-safe priority queue for spike events.
// Multiple producers (spike loop threads, gRPC input) can Push concurrently.
// Single consumer drains events for the current time step.
class EventQueue {
 public:
  // Push a single event (thread-safe).
  void Push(const SpikeEvent& event);

  // Push multiple events (thread-safe, single lock acquisition).
  void PushBatch(const std::vector<SpikeEvent>& events);

  // Drain all events with arrival_time < t_end into `out`.
  // Events are appended to `out` in arrival_time order.
  // Returns the number of events drained.
  int DrainUntil(float t_end, std::vector<SpikeEvent>& out);

  // Peek at the earliest event time without removing. Returns -1 if empty.
  float PeekTime() const;

  bool empty() const;
  size_t size() const;

 private:
  struct Node {
    SpikeEvent event;
    Node* next;
  };

  // Lock-free MPSC pending list. Producers push here; consumer drains into the
  // heap for time-ordered delivery.
  mutable std::atomic<Node*> pending_{nullptr};
  mutable std::atomic<size_t> pending_count_{0};

  using MinHeap = std::priority_queue<SpikeEvent, std::vector<SpikeEvent>,
                                      std::greater<SpikeEvent>>;
  mutable MinHeap heap_;  // single-consumer only

  // Move all pending nodes into the heap (consumer only).
  void DrainPendingToHeap() const;
};

}  // namespace senna::temporal

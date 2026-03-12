#include "core/temporal/event_queue.hpp"

namespace senna::temporal {

void EventQueue::Push(const SpikeEvent& event) {
  std::lock_guard lock(mutex_);
  heap_.push(event);
}

void EventQueue::PushBatch(const std::vector<SpikeEvent>& events) {
  std::lock_guard lock(mutex_);
  for (const auto& e : events) {
    heap_.push(e);
  }
}

int EventQueue::DrainUntil(float t_end, std::vector<SpikeEvent>& out) {
  std::lock_guard lock(mutex_);
  int count = 0;
  while (!heap_.empty() && heap_.top().arrival_time < t_end) {
    out.push_back(heap_.top());
    heap_.pop();
    ++count;
  }
  return count;
}

float EventQueue::PeekTime() const {
  std::lock_guard lock(mutex_);
  if (heap_.empty()) return -1.0f;
  return heap_.top().arrival_time;
}

bool EventQueue::empty() const {
  std::lock_guard lock(mutex_);
  return heap_.empty();
}

size_t EventQueue::size() const {
  std::lock_guard lock(mutex_);
  return heap_.size();
}

}  // namespace senna::temporal

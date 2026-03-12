#include "core/temporal/event_queue.hpp"

#include <vector>

namespace senna::temporal {

void EventQueue::Push(const SpikeEvent& event) {
  Node* node = new Node{event, nullptr};
  Node* old_head = pending_.load(std::memory_order_relaxed);
  do {
    node->next = old_head;
  } while (!pending_.compare_exchange_weak(
      old_head, node, std::memory_order_release, std::memory_order_relaxed));
  pending_count_.fetch_add(1, std::memory_order_relaxed);
}

void EventQueue::PushBatch(const std::vector<SpikeEvent>& events) {
  if (events.empty()) return;

  // Build a local LIFO list.
  Node* head = nullptr;
  for (const auto& e : events) {
    head = new Node{e, head};
  }

  // Attach list to pending stack.
  Node* old_head = pending_.load(std::memory_order_relaxed);
  do {
    // Find tail once per attempt.
    Node* tail = head;
    while (tail->next != nullptr) tail = tail->next;
    tail->next = old_head;
  } while (!pending_.compare_exchange_weak(
      old_head, head, std::memory_order_release, std::memory_order_relaxed));

  pending_count_.fetch_add(events.size(), std::memory_order_relaxed);
}

void EventQueue::DrainPendingToHeap() const {
  // Single consumer pops entire list.
  Node* list = pending_.exchange(nullptr, std::memory_order_acquire);
  size_t popped = 0;

  // Move to heap.
  for (Node* n = list; n != nullptr;) {
    heap_.push(n->event);
    Node* next = n->next;
    delete n;
    n = next;
    ++popped;
  }
  if (popped > 0) {
    pending_count_.fetch_sub(popped, std::memory_order_relaxed);
  }
}

int EventQueue::DrainUntil(float t_end, std::vector<SpikeEvent>& out) {
  DrainPendingToHeap();

  int drained = 0;
  while (!heap_.empty() && heap_.top().arrival_time < t_end) {
    out.push_back(heap_.top());
    heap_.pop();
    ++drained;
  }
  return drained;
}

float EventQueue::PeekTime() const {
  DrainPendingToHeap();
  if (heap_.empty()) return -1.0f;
  return heap_.top().arrival_time;
}

bool EventQueue::empty() const {
  DrainPendingToHeap();
  return heap_.empty();
}

size_t EventQueue::size() const {
  return heap_.size() + pending_count_.load(std::memory_order_relaxed);
}

}  // namespace senna::temporal

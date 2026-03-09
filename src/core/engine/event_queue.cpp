#include "core/engine/event_queue.h"

namespace senna::core::engine {

void EventQueue::clear() noexcept {
    queue_ = {};
    drained_.clear();
}

void EventQueue::push(const Event& event) { queue_.push(event); }

std::vector<EventQueue::Event> EventQueue::snapshot() const {
    std::vector<Event> events{};
    events.reserve(queue_.size());

    auto copy = queue_;
    while (!copy.empty()) {
        events.push_back(copy.top());
        copy.pop();
    }
    return events;
}

void EventQueue::restore(const std::vector<Event>& events) {
    clear();
    for (const auto& event : events) {
        push(event);
    }
}

const std::vector<EventQueue::Event>& EventQueue::drain_tick(
    const senna::core::domain::Time t_start, const senna::core::domain::Time t_end) {
    drained_.clear();
    if (t_end <= t_start) {
        return drained_;
    }

    while (!queue_.empty() && queue_.top().arrival < t_end) {
        auto event = queue_.top();
        queue_.pop();

        if (event.arrival >= t_start) {
            drained_.push_back(event);
        }
    }

    return drained_;
}

}  // namespace senna::core::engine

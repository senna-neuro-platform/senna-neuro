#pragma once

#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "core/domain/types.h"

namespace senna::core::engine {

class EventQueue final {
   public:
    using Event = senna::core::domain::SpikeEvent;

    [[nodiscard]] bool empty() const noexcept { return queue_.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return queue_.size(); }

    void clear() noexcept { queue_ = {}; }

    void push(Event event) { queue_.push(std::move(event)); }

    [[nodiscard]] std::vector<Event> snapshot() const {
        std::vector<Event> events{};
        events.reserve(queue_.size());

        auto copy = queue_;
        while (!copy.empty()) {
            events.push_back(copy.top());
            copy.pop();
        }
        return events;
    }

    void restore(const std::vector<Event>& events) {
        clear();
        for (const auto& event : events) {
            push(event);
        }
    }

    [[nodiscard]] std::vector<Event> drain_tick(const senna::core::domain::Time t_start,
                                                const senna::core::domain::Time t_end) {
        std::vector<Event> drained{};
        if (t_end <= t_start) {
            return drained;
        }

        while (!queue_.empty() && queue_.top().arrival < t_end) {
            auto event = queue_.top();
            queue_.pop();

            if (event.arrival >= t_start) {
                drained.push_back(event);
            }
        }

        return drained;
    }

   private:
    struct EarlierArrival {
        [[nodiscard]] bool operator()(const Event& lhs, const Event& rhs) const noexcept {
            return lhs.arrival > rhs.arrival;
        }
    };

    std::priority_queue<Event, std::vector<Event>, EarlierArrival> queue_{};
};

}  // namespace senna::core::engine

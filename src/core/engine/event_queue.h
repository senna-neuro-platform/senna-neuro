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

    void clear() noexcept;

    void push(const Event& event);

    [[nodiscard]] std::vector<Event> snapshot() const;

    void restore(const std::vector<Event>& events);

    [[nodiscard]] const std::vector<Event>& drain_tick(senna::core::domain::Time t_start,
                                                       senna::core::domain::Time t_end);

   private:
    struct EarlierArrival {
        [[nodiscard]] bool operator()(const Event& lhs, const Event& rhs) const noexcept {
            return lhs.arrival > rhs.arrival;
        }
    };

    std::priority_queue<Event, std::vector<Event>, EarlierArrival> queue_{};
    std::vector<Event> drained_{};
};

}  // namespace senna::core::engine

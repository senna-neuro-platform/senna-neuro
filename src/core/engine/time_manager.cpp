#include "core/engine/time_manager.h"

namespace senna::core::engine {

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
TimeManager::TimeManager(const senna::core::domain::Time dt, const senna::core::domain::Time start)
    : current_(start), dt_(dt) {
    if (dt_ <= 0.0F) {
        throw std::invalid_argument("Time step dt must be positive");
    }
}

void TimeManager::advance() noexcept { current_ += dt_; }

void TimeManager::reset(const senna::core::domain::Time start) noexcept { current_ = start; }

}  // namespace senna::core::engine

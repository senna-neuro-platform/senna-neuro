#pragma once

#include <stdexcept>

#include "core/domain/types.h"

namespace senna::core::engine {

class TimeManager final {
   public:
    explicit TimeManager(const senna::core::domain::Time dt = 0.5F,
                         const senna::core::domain::Time start = 0.0F)
        : current_(start), dt_(dt) {
        if (dt_ <= 0.0F) {
            throw std::invalid_argument("Time step dt must be positive");
        }
    }

    [[nodiscard]] senna::core::domain::Time elapsed() const noexcept { return current_; }

    [[nodiscard]] senna::core::domain::Time dt() const noexcept { return dt_; }

    void advance() noexcept { current_ += dt_; }

    void reset(const senna::core::domain::Time start = 0.0F) noexcept { current_ = start; }

   private:
    senna::core::domain::Time current_{0.0F};
    senna::core::domain::Time dt_{0.5F};
};

}  // namespace senna::core::engine

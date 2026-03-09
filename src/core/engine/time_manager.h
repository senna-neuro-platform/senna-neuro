#pragma once

#include <stdexcept>

#include "core/domain/types.h"

namespace senna::core::engine {

class TimeManager final {
   public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit TimeManager(senna::core::domain::Time dt = 0.5F,
                         senna::core::domain::Time start = 0.0F);

    [[nodiscard]] senna::core::domain::Time elapsed() const noexcept { return current_; }

    [[nodiscard]] senna::core::domain::Time dt() const noexcept { return dt_; }

    void advance() noexcept;

    void reset(senna::core::domain::Time start = 0.0F) noexcept;

   private:
    senna::core::domain::Time current_{0.0F};
    senna::core::domain::Time dt_{0.5F};
};

}  // namespace senna::core::engine

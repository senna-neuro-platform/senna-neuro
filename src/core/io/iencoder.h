#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/domain/types.h"

namespace senna::core::io {

using MnistImage = std::array<std::uint8_t, 28U * 28U>;

class IEncoder {
   public:
    virtual ~IEncoder() = default;

    [[nodiscard]] virtual std::vector<senna::core::domain::SpikeEvent> encode(
        const MnistImage& image, senna::core::domain::Time duration_ms) = 0;
};

}  // namespace senna::core::io

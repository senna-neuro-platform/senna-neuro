#pragma once

#include <vector>

#include "core/domain/types.h"

namespace senna::core::io {

class IDecoder {
   public:
    virtual ~IDecoder() = default;

    [[nodiscard]] virtual int decode(
        const std::vector<senna::core::domain::SpikeEvent>& output_spikes) const = 0;
};

}  // namespace senna::core::io

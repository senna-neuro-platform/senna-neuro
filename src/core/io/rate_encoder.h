#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/domain/lattice.h"
#include "core/io/iencoder.h"

namespace senna::core::io {

struct RateEncoderConfig {
    float max_rate_hz{100.0F};
    senna::core::domain::Time dt{0.5F};
    senna::core::domain::Weight spike_value{1.1F};
    std::uint32_t seed{42U};
};

class RateEncoder final : public IEncoder {
   public:
    explicit RateEncoder(const senna::core::domain::Lattice& lattice,
                         RateEncoderConfig config = {});

    [[nodiscard]] std::vector<senna::core::domain::SpikeEvent> encode(
        const MnistImage& image, senna::core::domain::Time duration_ms) override;

   private:
    static constexpr std::size_t kImageWidth = 28U;
    static constexpr std::size_t kImageHeight = 28U;
    static constexpr std::size_t kImageSize = kImageWidth * kImageHeight;

    RateEncoderConfig config_{};
    std::mt19937 random_{};
    std::uniform_real_distribution<float> unit_dist_{};
    std::vector<senna::core::domain::NeuronId> sensor_map_{};
};

}  // namespace senna::core::io

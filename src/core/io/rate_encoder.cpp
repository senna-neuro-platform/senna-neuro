#include "core/io/rate_encoder.h"

namespace senna::core::io {

RateEncoder::RateEncoder(const senna::core::domain::Lattice& lattice,
                         const RateEncoderConfig config)
    : config_(config), random_(config.seed), unit_dist_(0.0F, 1.0F) {
    if (config_.max_rate_hz < 0.0F) {
        throw std::invalid_argument("RateEncoderConfig.max_rate_hz must be non-negative");
    }
    if (config_.dt <= 0.0F) {
        throw std::invalid_argument("RateEncoderConfig.dt must be positive");
    }

    sensor_map_.reserve(kImageSize);
    for (std::uint16_t y = 0U; y < kImageHeight; ++y) {
        for (std::uint16_t x = 0U; x < kImageWidth; ++x) {
            const auto id = lattice.neuron_id_at(senna::core::domain::Coord3D{x, y, 0U});
            if (!id.has_value()) {
                throw std::invalid_argument("RateEncoder requires fully populated sensor layer");
            }
            sensor_map_.push_back(*id);
        }
    }
}

std::vector<senna::core::domain::SpikeEvent> RateEncoder::encode(
    const MnistImage& image, const senna::core::domain::Time duration_ms) {
    std::vector<senna::core::domain::SpikeEvent> spikes{};
    if (duration_ms <= 0.0F) {
        return spikes;
    }

    const auto ticks = static_cast<std::size_t>(duration_ms / config_.dt);
    spikes.reserve(ticks * 4U);

    for (std::size_t tick = 0U; tick < ticks; ++tick) {
        const auto t = static_cast<senna::core::domain::Time>(tick) * config_.dt;

        for (std::size_t idx = 0U; idx < kImageSize; ++idx) {
            const auto pixel = image[idx];
            if (pixel == 0U) {
                continue;
            }

            const auto rate = (static_cast<float>(pixel) / 255.0F) * config_.max_rate_hz;
            const auto probability = std::clamp((rate * config_.dt) / 1000.0F, 0.0F, 1.0F);
            if (unit_dist_(random_) < probability) {
                const auto target = sensor_map_[idx];
                spikes.push_back(senna::core::domain::SpikeEvent{
                    target,
                    target,
                    t,
                    config_.spike_value,
                });
            }
        }
    }

    return spikes;
}

}  // namespace senna::core::io

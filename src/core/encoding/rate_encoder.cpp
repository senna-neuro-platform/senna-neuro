#include "core/encoding/rate_encoder.hpp"

namespace senna::encoding {

RateEncoder::RateEncoder(const RateEncoderParams& params, float dt,
                         uint64_t seed)
    : params_(params), dt_(dt), rng_(seed) {}

int RateEncoder::Encode(std::span<const uint8_t> image,
                        const spatial::ZonedLattice& lattice,
                        temporal::EventQueue& queue, float t_start) {
  int width = lattice.width();
  int height = lattice.height();
  int total = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      if (idx >= static_cast<int>(image.size())) {
        break;
      }
      total += EncodeSinglePixel(x, y, image[idx], lattice, queue, t_start);
    }
  }
  return total;
}

int RateEncoder::EncodeSinglePixel(int x, int y, uint8_t intensity,
                                   const spatial::ZonedLattice& lattice,
                                   temporal::EventQueue& queue, float t_start) {
  auto neuron_id = lattice.SensoryNeuron(x, y);
  if (neuron_id == spatial::kEmptyVoxel) {
    return 0;
  }

  float rate = (static_cast<float>(intensity) / 255.0F) * params_.max_rate;
  float p_spike = rate * dt_ / 1000.0F;  // probability per tick

  std::uniform_real_distribution<float> dist(0.0F, 1.0F);

  int count = 0;
  int num_ticks = static_cast<int>(params_.presentation_ms / dt_);

  for (int tick = 0; tick < num_ticks; ++tick) {
    if (dist(rng_) < p_spike) {
      float t = t_start + static_cast<float>(tick) * dt_;
      queue.Push({.target_id = neuron_id,
                  .source_id = -1,
                  .arrival_time = t,
                  .value = params_.input_value});
      ++count;
    }
  }
  return count;
}

}  // namespace senna::encoding

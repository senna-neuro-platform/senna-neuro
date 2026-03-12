#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "core/spatial/lattice.hpp"
#include "core/temporal/event_queue.hpp"

namespace senna::encoding {

// Parameters for rate coding.
struct RateEncoderParams {
  float max_rate = 100.0f;        // Hz (for pixel value 255)
  float presentation_ms = 50.0f;  // presentation duration (ms)
  float input_value = 1.5f;       // spike event value injected into neuron
};

inline constexpr RateEncoderParams kDefaultEncoderParams{};

// Converts a 28x28 MNIST image into a Poisson spike train on the sensory panel.
//
// For each pixel (x, y) with intensity [0, 255]:
//   rate = (intensity / 255.0) * max_rate
//   For each dt tick: P(spike) = rate * dt / 1000
//   If spike -> push SpikeEvent to target neuron at (x, y, 0)
class RateEncoder {
 public:
  // dt: time step (ms), same as TimeManager dt.
  RateEncoder(const RateEncoderParams& params = kDefaultEncoderParams,
              float dt = 0.5f, uint64_t seed = 42);

  // Encode one image and push spike events into the queue.
  // image: 28x28 = 784 pixel values (row-major, [0..255]).
  // t_start: virtual time when presentation begins.
  // Returns total number of spike events generated.
  int Encode(std::span<const uint8_t> image,
             const spatial::ZonedLattice& lattice, temporal::EventQueue& queue,
             float t_start);

  // Encode a single pixel at (x, y) with given intensity.
  // Useful for testing.
  int EncodeSinglePixel(int x, int y, uint8_t intensity,
                        const spatial::ZonedLattice& lattice,
                        temporal::EventQueue& queue, float t_start);

 private:
  RateEncoderParams params_;
  float dt_;
  std::mt19937_64 rng_;
};

}  // namespace senna::encoding

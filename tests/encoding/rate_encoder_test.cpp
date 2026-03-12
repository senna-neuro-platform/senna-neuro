#include "core/encoding/rate_encoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace senna::encoding {
namespace {

class RateEncoderTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSeed = 42;
  static constexpr int kW = 28;
  static constexpr int kH = 28;

  spatial::ZonedLattice lattice_{kW, kH, 5, 0.7, kSeed};

  // Helper: drain all events and count them.
  int DrainAll(temporal::EventQueue& queue) {
    std::vector<temporal::SpikeEvent> out;
    queue.DrainUntil(1e6f, out);
    return static_cast<int>(out.size());
  }

  int EncodeAndCount(const std::vector<uint8_t>& image,
                     uint64_t seed_offset = 0) {
    RateEncoder enc({}, 0.5f, kSeed + seed_offset);
    temporal::EventQueue queue;
    enc.Encode(image, lattice_, queue, 0.0f);
    return DrainAll(queue);
  }
};

TEST_F(RateEncoderTest, WhitePixelHighRate) {
  RateEncoder enc({}, 0.5f, kSeed);
  temporal::EventQueue queue;

  // White pixel (255) for 50ms at 100Hz -> expect ~5 spikes.
  int count = enc.EncodeSinglePixel(0, 0, 255, lattice_, queue, 0.0f);

  // With Poisson rate 100Hz, 50ms: expected ~5, allow 2-8.
  EXPECT_GE(count, 2);
  EXPECT_LE(count, 8);
  EXPECT_EQ(static_cast<int>(queue.size()), count);
}

TEST_F(RateEncoderTest, BlackPixelNoSpikes) {
  RateEncoder enc({}, 0.5f, kSeed);
  temporal::EventQueue queue;

  int count = enc.EncodeSinglePixel(0, 0, 0, lattice_, queue, 0.0f);
  EXPECT_EQ(count, 0);
  EXPECT_TRUE(queue.empty());
}

TEST_F(RateEncoderTest, MediumPixelModerateRate) {
  // Run multiple trials to get statistical average.
  int total = 0;
  int trials = 20;
  for (int i = 0; i < trials; ++i) {
    RateEncoder enc({}, 0.5f, kSeed + i);
    temporal::EventQueue queue;
    total += enc.EncodeSinglePixel(0, 0, 128, lattice_, queue, 0.0f);
  }
  float avg = static_cast<float>(total) / trials;
  // 128/255 * 100Hz * 50ms/1000 = ~2.5 spikes expected.
  EXPECT_GE(avg, 1.0f);
  EXPECT_LE(avg, 4.0f);
}

TEST_F(RateEncoderTest, BrighterImageMoreSpikes) {
  // All-white image vs all-dark (64) image.
  std::vector<uint8_t> bright(784, 255);
  std::vector<uint8_t> dark(784, 64);

  RateEncoder enc1({}, 0.5f, kSeed);
  temporal::EventQueue q1;
  int bright_count = enc1.Encode(bright, lattice_, q1, 0.0f);

  RateEncoder enc2({}, 0.5f, kSeed);
  temporal::EventQueue q2;
  int dark_count = enc2.Encode(dark, lattice_, q2, 0.0f);

  EXPECT_GT(bright_count, dark_count);
}

TEST_F(RateEncoderTest, TotalSpikesProportionalToBrightness) {
  // Image A: all 200; Image B: all 50 (4x brightness ratio).
  std::vector<uint8_t> img_a(784, 200);
  std::vector<uint8_t> img_b(784, 50);

  // Average over multiple seeds to smooth randomness.
  float avg_a = 0.0f, avg_b = 0.0f;
  int trials = 8;
  for (int i = 0; i < trials; ++i) {
    avg_a += EncodeAndCount(img_a, i);
    avg_b += EncodeAndCount(img_b, i + 100);
  }
  avg_a /= trials;
  avg_b /= trials;

  // Expect roughly 4x spikes; allow slack due to Poisson variance.
  EXPECT_GT(avg_a, 2.5f * avg_b);
}

TEST_F(RateEncoderTest, DifferentImagesProduceDifferentTargets) {
  // Single bright pixel in different locations should generate events to
  // distinct neurons.
  std::vector<uint8_t> img_a(784, 0);
  std::vector<uint8_t> img_b(784, 0);
  img_a[0] = 255;             // (0,0)
  img_b[27 * 28 + 27] = 255;  // (27,27)

  RateEncoder enc_a({}, 0.5f, kSeed);
  RateEncoder enc_b({}, 0.5f,
                    kSeed);  // same seed; RNG path differs by location
  temporal::EventQueue q1, q2;

  enc_a.Encode(img_a, lattice_, q1, 0.0f);
  enc_b.Encode(img_b, lattice_, q2, 0.0f);

  std::vector<temporal::SpikeEvent> e1, e2;
  q1.DrainUntil(1e6f, e1);
  q2.DrainUntil(1e6f, e2);

  // They may have different counts; require at least one differing target id.
  bool targets_differ = false;
  size_t n = std::min(e1.size(), e2.size());
  for (size_t i = 0; i < n; ++i) {
    if (e1[i].target_id != e2[i].target_id) {
      targets_differ = true;
      break;
    }
  }
  if (!targets_differ && !e1.empty() && !e2.empty() && e1.size() != e2.size()) {
    targets_differ = true;
  }

  EXPECT_TRUE(targets_differ);
}

TEST_F(RateEncoderTest, HigherIntensityYieldsMoreSpikesSinglePixel) {
  RateEncoder enc({}, 0.5f, kSeed);
  temporal::EventQueue q1, q2;
  int c1 = enc.EncodeSinglePixel(0, 0, 200, lattice_, q1, 0.0f);
  // Rerun with a fresh encoder/seed to avoid RNG depletion.
  RateEncoder enc2({}, 0.5f, kSeed);
  int c2 = enc2.EncodeSinglePixel(0, 0, 50, lattice_, q2, 0.0f);
  EXPECT_GT(c1, c2);
}

TEST_F(RateEncoderTest, FullImageEncode) {
  // Encode a checkerboard pattern.
  std::vector<uint8_t> image(784, 0);
  for (int i = 0; i < 784; i += 2) {
    image[i] = 200;
  }

  RateEncoder enc({}, 0.5f, kSeed);
  temporal::EventQueue queue;
  int count = enc.Encode(image, lattice_, queue, 0.0f);

  // Half pixels at 200, half at 0. Should produce spikes.
  EXPECT_GT(count, 0);
  EXPECT_EQ(static_cast<int>(queue.size()), count);
}

TEST_F(RateEncoderTest, Determinism) {
  std::vector<uint8_t> image(784);
  for (int i = 0; i < 784; ++i) image[i] = static_cast<uint8_t>(i % 256);

  RateEncoder enc1({}, 0.5f, kSeed);
  temporal::EventQueue q1;
  int c1 = enc1.Encode(image, lattice_, q1, 0.0f);

  RateEncoder enc2({}, 0.5f, kSeed);
  temporal::EventQueue q2;
  int c2 = enc2.Encode(image, lattice_, q2, 0.0f);

  EXPECT_EQ(c1, c2);

  // Drain and compare events.
  std::vector<temporal::SpikeEvent> e1, e2;
  q1.DrainUntil(1e6f, e1);
  q2.DrainUntil(1e6f, e2);

  ASSERT_EQ(e1.size(), e2.size());
  for (size_t i = 0; i < e1.size(); ++i) {
    EXPECT_EQ(e1[i].target_id, e2[i].target_id);
    EXPECT_FLOAT_EQ(e1[i].arrival_time, e2[i].arrival_time);
  }
}

TEST_F(RateEncoderTest, EventTimesWithinPresentation) {
  std::vector<uint8_t> image(784, 200);

  RateEncoder enc({}, 0.5f, kSeed);
  temporal::EventQueue queue;
  enc.Encode(image, lattice_, queue, 10.0f);

  std::vector<temporal::SpikeEvent> events;
  queue.DrainUntil(1e6f, events);

  for (const auto& e : events) {
    EXPECT_GE(e.arrival_time, 10.0f);
    EXPECT_LT(e.arrival_time, 60.0f);  // t_start + presentation_ms
  }
}

}  // namespace
}  // namespace senna::encoding

#include "core/decoding/first_spike_decoder.hpp"

#include <gtest/gtest.h>

namespace senna::decoding {
namespace {

TEST(FirstSpikeDecoderTest, PicksFirstOutputSpike) {
  FirstSpikeDecoder dec({10, 20, 30});
  dec.Observe(5, 1.0f);   // ignore non-output
  dec.Observe(20, 2.0f);  // winner
  dec.Observe(10, 2.5f);  // later than winner, ignored

  ASSERT_TRUE(dec.Result().has_value());
  EXPECT_EQ(dec.Result().value(), 1);  // index of neuron 20
}

TEST(FirstSpikeDecoderTest, NoSpikesYieldsEmpty) {
  FirstSpikeDecoder dec({10, 11});
  EXPECT_FALSE(dec.Result().has_value());
}

TEST(FirstSpikeDecoderTest, ResetClearsWinner) {
  FirstSpikeDecoder dec({1, 2});
  dec.Observe(2, 0.1f);
  ASSERT_TRUE(dec.Result().has_value());
  dec.Reset(0.0f);
  EXPECT_FALSE(dec.Result().has_value());
  dec.Observe(1, 0.2f);
  EXPECT_EQ(dec.Result().value(), 0);
}

TEST(FirstSpikeDecoderTest, TimeoutYieldsUndefinedAndIgnoresLateSpikes) {
  FirstSpikeDecoder dec({10, 11}, /*window_ms=*/50.0f);
  dec.SetStartTime(0.0f);

  // No spikes within the window.
  EXPECT_FALSE(dec.ResultWithTimeout(49.0f).has_value());
  EXPECT_FALSE(dec.ResultWithTimeout(51.0f).has_value());

  // Late spike after window should be ignored.
  dec.Observe(10, 55.0f);
  EXPECT_FALSE(dec.Result().has_value());
  EXPECT_FALSE(dec.ResultWithTimeout(60.0f).has_value());
}

TEST(FirstSpikeDecoderTest, SpikeBeforeTimeoutWins) {
  FirstSpikeDecoder dec({5, 6}, /*window_ms=*/50.0f);
  dec.SetStartTime(0.0f);
  dec.Observe(6, 20.0f);

  auto result = dec.ResultWithTimeout(30.0f);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 1);
}

TEST(FirstSpikeDecoderTest, StochasticTieBreakDeterministicWithSeed) {
  FirstSpikeDecoder dec({10, 20}, 50.0f, /*seed=*/123);
  dec.SetStartTime(0.0f);
  dec.Observe(10, 5.0f);
  dec.Observe(20, 5.0f);  // same timestamp -> tie
  dec.Finalize(5.1f);
  ASSERT_TRUE(dec.Result().has_value());
  int first = dec.Result().value();

  // Run again with same seed to ensure deterministic tie-break.
  FirstSpikeDecoder dec2({10, 20}, 50.0f, /*seed=*/123);
  dec2.SetStartTime(0.0f);
  dec2.Observe(10, 5.0f);
  dec2.Observe(20, 5.0f);
  dec2.Finalize(5.1f);
  ASSERT_TRUE(dec2.Result().has_value());
  EXPECT_EQ(first, dec2.Result().value());
}

}  // namespace
}  // namespace senna::decoding

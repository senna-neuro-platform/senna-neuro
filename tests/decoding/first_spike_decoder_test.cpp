#include "core/decoding/first_spike_decoder.hpp"

#include <gtest/gtest.h>

namespace senna::decoding {
namespace {

TEST(FirstSpikeDecoderTest, PicksFirstOutputSpike) {
  FirstSpikeDecoder dec({10, 20, 30});
  dec.Observe(5, 1.0f);   // ignore non-output
  dec.Observe(20, 2.0f);  // winner
  dec.Observe(10, 1.5f);  // later than winner, but after decision ignored

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

}  // namespace
}  // namespace senna::decoding

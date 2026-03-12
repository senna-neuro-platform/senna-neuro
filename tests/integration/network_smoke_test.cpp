#include <gtest/gtest.h>

#include <set>

#include "core/network/network_builder.hpp"
#include "core/network/spike_loop.hpp"

namespace senna::network {
namespace {

// Small network for integration tests.
NetworkConfig SmallConfig() {
  return {
      .width = 10,
      .height = 10,
      .depth = 5,
      .density = 0.8,
      .neighbor_radius = 2.0f,
      .excitatory_ratio = 0.8,
      .num_outputs = 3,
      .dt = 0.5f,
      .seed = 42,
  };
}

TEST(NetworkSmokeTest, ConstructionSucceeds) {
  Network net(SmallConfig());
  EXPECT_GT(net.pool().size(), 0);
  EXPECT_GT(net.synapses().synapse_count(), 0);
  EXPECT_FLOAT_EQ(net.time_manager().time(), 0.0f);
}

TEST(NetworkSmokeTest, NoStimulusSilence) {
  Network net(SmallConfig());
  SpikeLoop loop(net);

  auto stats = loop.Run(10.0f);  // 10 ms, no stimulus
  EXPECT_EQ(stats.total_spikes, 0);
  EXPECT_EQ(stats.active_neurons, 0);
  EXPECT_FLOAT_EQ(net.time_manager().time(), 10.0f);
}

TEST(NetworkSmokeTest, SingleStimulusWave) {
  Network net(SmallConfig());
  SpikeLoop loop(net);

  // Inject a strong spike into neuron 0.
  net.InjectSpike(0, 0.1f, 1.5f);

  auto stats = loop.Run(20.0f);

  // At least the stimulated neuron should fire.
  EXPECT_GE(stats.total_spikes, 1);
  EXPECT_GE(stats.active_neurons, 1);
}

TEST(NetworkSmokeTest, MultipleStimuliMoreSpikes) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  // Inject spikes into 5 sensory neurons.
  for (int i = 0; i < 5; ++i) {
    net.InjectSensory(i, 0, 0.1f, 1.5f);
  }

  auto stats = loop.Run(20.0f);
  EXPECT_GE(stats.total_spikes, 5);
}

TEST(NetworkSmokeTest, SensoryInjection) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  // Inject into sensory panel at (5, 5).
  net.InjectSensory(5, 5, 0.1f, 1.5f);

  auto stats = loop.Run(10.0f);
  EXPECT_GE(stats.total_spikes, 1);
}

TEST(NetworkSmokeTest, EncodedImageProducesSpikes) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  // Bright 10x10 patch encoded at t=0.1.
  std::vector<uint8_t> image(config.width * config.height, 0);
  for (int i = 0; i < 50; ++i) image[i] = 255;

  net.EncodeImage(image, 0.1f);

  auto stats = loop.Run(20.0f);
  EXPECT_GT(stats.total_spikes, 0);
}

TEST(NetworkSmokeTest, Determinism) {
  auto config = SmallConfig();

  // Run 1
  Network net1(config);
  SpikeLoop loop1(net1);
  net1.InjectSpike(0, 0.1f, 1.5f);
  auto stats1 = loop1.Run(10.0f);

  // Run 2 (same seed)
  Network net2(config);
  SpikeLoop loop2(net2);
  net2.InjectSpike(0, 0.1f, 1.5f);
  auto stats2 = loop2.Run(10.0f);

  EXPECT_EQ(stats1.total_spikes, stats2.total_spikes);
  EXPECT_EQ(stats1.active_neurons, stats2.active_neurons);

  // Spike logs should be identical.
  ASSERT_EQ(loop1.spike_log().size(), loop2.spike_log().size());
  for (size_t i = 0; i < loop1.spike_log().size(); ++i) {
    EXPECT_EQ(loop1.spike_log()[i].first, loop2.spike_log()[i].first);
    EXPECT_FLOAT_EQ(loop1.spike_log()[i].second, loop2.spike_log()[i].second);
  }
}

TEST(NetworkSmokeTest, SpikeLogRecordsNeuronAndTime) {
  Network net(SmallConfig());
  SpikeLoop loop(net);
  net.InjectSpike(0, 0.1f, 1.5f);
  loop.Run(10.0f);

  for (const auto& [id, time] : loop.spike_log()) {
    EXPECT_GE(id, 0);
    EXPECT_LT(id, net.pool().size());
    EXPECT_GE(time, 0.0f);
    EXPECT_LE(time, 10.0f);
  }
}

}  // namespace
}  // namespace senna::network

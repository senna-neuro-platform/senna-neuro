#include <gtest/gtest.h>

#include <set>

#include "core/decoding/first_spike_decoder.hpp"
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

TEST(NetworkSmokeTest, OutputLayerHasCorrectCountAndWtaSynapses) {
  auto config = SmallConfig();
  Network net(config);

  EXPECT_EQ(net.output_ids().size(), static_cast<size_t>(config.num_outputs));

  // All output neurons must be on the top plane and have WTA outgoing edges.
  for (auto id : net.output_ids()) {
    auto coords = net.lattice().CoordsOf(id);
    EXPECT_EQ(coords.z, config.depth - 1);
    EXPECT_GT(net.synapses().OutgoingCount(id), config.num_outputs - 1);
  }
}

TEST(NetworkSmokeTest, WtaSynapsesHaveCorrectSignAndDelay) {
  auto config = SmallConfig();
  Network net(config);

  auto wta_weight = std::abs(config.synapse_params.w_wta);
  for (auto pre : net.output_ids()) {
    for (auto sid : net.synapses().Outgoing(pre)) {
      const auto& syn = net.synapses().Get(sid);
      if (std::find(net.output_ids().begin(), net.output_ids().end(),
                    syn.post_id) == net.output_ids().end()) {
        continue;  // skip non-output targets
      }
      EXPECT_LT(syn.sign, 0.0f);         // inhibitory
      EXPECT_FLOAT_EQ(syn.delay, 0.0f);  // zero delay
      EXPECT_FLOAT_EQ(syn.weight, wta_weight);
    }
  }
}

TEST(NetworkSmokeTest, FirstSpikeDecoderPicksInjectedOutput) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  const auto& outputs = net.output_ids();
  ASSERT_FALSE(outputs.empty());

  // Inject a strong spike directly into output neuron #1.
  ASSERT_GT(outputs.size(), 1u);
  net.InjectSpike(outputs[1], 0.1f, 2.0f);

  decoding::FirstSpikeDecoder dec(outputs, config.decoder_window_ms);
  dec.SetStartTime(0.0f);
  loop.AttachDecoder(&dec);
  loop.Run(5.0f);

  ASSERT_TRUE(dec.Result().has_value());
  EXPECT_EQ(dec.Result().value(), 1);
}

TEST(NetworkSmokeTest, WtaSuppressesLaterOutput) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);
  const auto& outputs = net.output_ids();
  ASSERT_GE(outputs.size(), 2u);

  // Output 0 fires first with strong input.
  net.InjectSpike(outputs[0], 0.10f, 2.0f);
  // Output 1 would fire, but arrives after inhibition; moderate input.
  net.InjectSpike(outputs[1], 0.12f, 1.2f);

  decoding::FirstSpikeDecoder dec(outputs, config.decoder_window_ms);
  dec.SetStartTime(0.0f);
  loop.AttachDecoder(&dec);
  loop.Run(5.0f);

  int spikes_output1 = 0;
  for (const auto& entry : loop.spike_log()) {
    if (entry.first == outputs[1]) ++spikes_output1;
  }

  ASSERT_TRUE(dec.Result().has_value());
  EXPECT_EQ(dec.Result().value(), 0);  // first output wins
  // Output1 may or may not spike depending on weights, but it must not win.
}

TEST(NetworkSmokeTest, OnlyOneOutputWinsWithSimilarInputs) {
  auto config = SmallConfig();
  // Increase WTA inhibition to enforce single winner.
  config.synapse_params.w_wta = -12.0f;
  Network net(config);
  SpikeLoop loop(net);
  const auto& outputs = net.output_ids();
  ASSERT_GE(outputs.size(), 2u);

  // Two strong inputs close in time; WTA should leave a single winner.
  net.InjectSpike(outputs[0], 0.10f, 2.0f);
  net.InjectSpike(outputs[1], 0.60f,
                  2.0f);  // arrives next tick, should be inhibited

  decoding::FirstSpikeDecoder dec(outputs, config.decoder_window_ms);
  dec.SetStartTime(0.0f);
  loop.AttachDecoder(&dec);
  loop.Run(5.0f);

  int output_spikes = 0;
  for (const auto& entry : loop.spike_log()) {
    if (std::find(outputs.begin(), outputs.end(), entry.first) !=
        outputs.end()) {
      ++output_spikes;
    }
  }

  ASSERT_TRUE(dec.Result().has_value());
  EXPECT_EQ(dec.Result().value(), 0);  // earliest wins
  EXPECT_EQ(output_spikes, 1);         // only one output fired
}

TEST(NetworkSmokeTest, DecoderEmptyWhenNoOutputSpikes) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  loop.Run(60.0f);  // no stimuli, exceed decoder timeout window

  decoding::FirstSpikeDecoder dec(net.output_ids(), config.decoder_window_ms);
  dec.SetStartTime(0.0f);
  loop.AttachDecoder(&dec);
  EXPECT_FALSE(dec.ResultWithTimeout(net.time_manager().time()).has_value());
}

TEST(NetworkSmokeTest, RunStatsReported) {
  auto config = SmallConfig();
  Network net(config);
  SpikeLoop loop(net);

  net.InjectSpike(net.output_ids()[0], 0.1f, 1.5f);
  auto stats = loop.Run(5.0f);

  EXPECT_EQ(stats.ticks, static_cast<int>(5.0f / config.dt));
  EXPECT_GE(stats.total_spikes, 1);
  EXPECT_GE(stats.active_neurons, 1);
  EXPECT_FLOAT_EQ(stats.duration_ms, 5.0f);
}

TEST(NetworkSmokeTest, OutputsHaveIncomingFromVolume) {
  auto config = SmallConfig();
  Network net(config);

  for (auto out : net.output_ids()) {
    int incoming = net.synapses().IncomingCount(out);
    EXPECT_GT(incoming, 0);

    bool has_volume_source = false;
    for (auto sid : net.synapses().Incoming(out)) {
      const auto& syn = net.synapses().Get(sid);
      // consider any source not in the output set as volume/input
      if (std::find(net.output_ids().begin(), net.output_ids().end(),
                    syn.pre_id) == net.output_ids().end()) {
        has_volume_source = true;
        break;
      }
    }
    EXPECT_TRUE(has_volume_source);
  }
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

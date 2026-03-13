#include "core/plasticity/stdp.hpp"

#include <gtest/gtest.h>

#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"

namespace senna::plasticity {
namespace {

struct SmallNet {
  spatial::Lattice lattice{2, 1, 1, 1.0, 42};
  spatial::NeighborIndex neighbors{lattice, 2.0F, 1};
  neural::NeuronPool pool{lattice, neural::kDefaultLIF, 1.0, 123};
  synaptic::SynapseIndex syn_index{lattice, neighbors, pool};
  synaptic::SynapseParams syn_params{};
};

TEST(STDPTest, CausalPairIncreasesWeight) {
  SmallNet net;
  auto sid = net.syn_index.Outgoing(0)[0];
  float w0 = net.syn_index.Get(sid).weight;

  net.pool.Fire(0, 10.0F);  // pre
  net.pool.Fire(1, 15.0F);  // post

  STDP::OnPostSpike(1, 15.0F, net.syn_index, net.pool, net.syn_params);

  float w1 = net.syn_index.Get(sid).weight;
  EXPECT_GT(w1, w0);
}

TEST(STDPTest, AntiCausalPairDecreasesWeight) {
  SmallNet net;
  auto sid = net.syn_index.Outgoing(0)[0];
  net.syn_index.Get(sid).weight = 0.08F;

  net.pool.Fire(1, 10.0F);  // post before pre
  net.pool.Fire(0, 15.0F);  // pre

  STDP::OnPreSpike(0, 15.0F, net.syn_index, net.pool, net.syn_params);

  float w1 = net.syn_index.Get(sid).weight;
  EXPECT_LT(w1, 0.08F);
  EXPECT_GE(w1, net.syn_params.w_min);
}

TEST(STDPTest, LargeTimeGapHasTinyEffect) {
  SmallNet net;
  auto sid = net.syn_index.Outgoing(0)[0];
  float w0 = net.syn_index.Get(sid).weight;

  net.pool.Fire(0, 0.0F);
  net.pool.Fire(1, 100.0F);  // big gap

  STDP::OnPostSpike(1, 100.0F, net.syn_index, net.pool, net.syn_params);

  float w1 = net.syn_index.Get(sid).weight;
  EXPECT_NEAR(w1, w0, 1e-3F);
}

TEST(STDPTest, NoUpdateIfPartnerNeverSpiked) {
  // Case 1: pre fires, post never fired.
  {
    SmallNet net;
    auto sid = net.syn_index.Outgoing(0)[0];
    float w0 = net.syn_index.Get(sid).weight;
    net.pool.Fire(0, 10.0F);
    STDP::OnPreSpike(0, 10.0F, net.syn_index, net.pool, net.syn_params);
    EXPECT_FLOAT_EQ(net.syn_index.Get(sid).weight, w0);
  }

  // Case 2: post fires, pre never fired.
  {
    SmallNet net;
    auto sid = net.syn_index.Outgoing(0)[0];
    float w0 = net.syn_index.Get(sid).weight;
    net.pool.Fire(1, 20.0F);
    STDP::OnPostSpike(1, 20.0F, net.syn_index, net.pool, net.syn_params);
    EXPECT_FLOAT_EQ(net.syn_index.Get(sid).weight, w0);
  }
}

TEST(STDPTest, PotentiationTapersNearWmax) {
  SmallNet net;
  auto sid = net.syn_index.Outgoing(0)[0];
  net.syn_index.Get(sid).weight = 0.99F;  // close to w_max=1.0

  net.pool.Fire(0, 0.0F);
  net.pool.Fire(1, 1.0F);

  STDP::OnPostSpike(1, 1.0F, net.syn_index, net.pool, net.syn_params,
                    STDPParams{});

  // Weight should increase only slightly and stay <= w_max.
  float w1 = net.syn_index.Get(sid).weight;
  EXPECT_LE(w1, 1.0F);
  EXPECT_GT(w1, 0.99F);
  EXPECT_LT(w1 - 0.99F, 0.01F);  // taper reduced the jump
}

TEST(STDPTest, SupervisionForcesPostSpikeAndPotentiates) {
  SmallNet net;
  auto sid = net.syn_index.Outgoing(0)[0];
  net.syn_index.Get(sid).weight = 0.05F;

  // Pre fires during stimulus.
  net.pool.Fire(0, 0.0F);

  // Supervise correct output neuron.
  STDP::Supervise(1, 5.0F, net.syn_index, net.pool, net.syn_params,
                  STDPParams{});

  float w1 = net.syn_index.Get(sid).weight;
  EXPECT_GT(w1, 0.05F);
}

}  // namespace
}  // namespace senna::plasticity

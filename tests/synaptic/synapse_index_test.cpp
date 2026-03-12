#include "core/synaptic/synapse_index.hpp"

#include <gtest/gtest.h>

namespace senna::synaptic {
namespace {

TEST(SynapseIndexTest, DelaysFollowDistance) {
  spatial::Lattice lattice(3, 3, 1, 1.0, 42);
  spatial::NeighborIndex neighbors(lattice, 1.5f, 1);
  neural::NeuronPool pool(lattice, neural::kDefaultLIF, 1.0, 123);

  SynapseIndex idx(lattice, neighbors, pool);
  ASSERT_GT(idx.synapse_count(), 0);

  const auto& syn = idx.Get(0);
  auto pre = lattice.CoordsOf(syn.pre_id);
  auto post = lattice.CoordsOf(syn.post_id);
  float dx = pre.x - post.x;
  float dy = pre.y - post.y;
  float dz = pre.z - post.z;
  float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
  EXPECT_NEAR(syn.delay, dist * kDefaultSynapseParams.c_base, 1e-5f);
}

TEST(SynapseIndexTest, WeightWithinRangeAndSignMatchesType) {
  spatial::Lattice lattice(3, 3, 1, 1.0, 42);
  spatial::NeighborIndex neighbors(lattice, 1.5f, 1);

  // All inhibitory to test sign.
  neural::NeuronPool pool(lattice, neural::kDefaultLIF, 0.0, 321);
  SynapseIndex idx(lattice, neighbors, pool);

  for (int32_t i = 0; i < idx.synapse_count(); ++i) {
    const auto& syn = idx.Get(i);
    EXPECT_LE(syn.weight, kDefaultSynapseParams.w_max);
    EXPECT_GE(syn.weight, kDefaultSynapseParams.w_min);
    EXPECT_LT(syn.sign, 0.0f);
  }
}

TEST(SynapseIndexTest, WtaConnectionsAddedWithCorrectParams) {
  spatial::Lattice lattice(3, 3, 1, 1.0, 42);
  spatial::NeighborIndex neighbors(lattice, 1.5f, 1);
  neural::NeuronPool pool(lattice, neural::kDefaultLIF, 1.0, 123);

  std::vector<int32_t> outputs = {0, 1, 2};
  SynapseParams params = kDefaultSynapseParams;
  params.w_wta = -3.0f;
  SynapseIndex idx(lattice, neighbors, pool, outputs, params, 7);

  EXPECT_EQ(idx.wta_count(),
            static_cast<int32_t>(outputs.size() * (outputs.size() - 1)));

  for (int32_t i = idx.synapse_count() - idx.wta_count();
       i < idx.synapse_count(); ++i) {
    const auto& syn = idx.Get(i);
    EXPECT_EQ(syn.delay, 0.0f);
    EXPECT_FLOAT_EQ(syn.weight, std::abs(params.w_wta));
    EXPECT_LT(syn.sign, 0.0f);
  }
}

}  // namespace
}  // namespace senna::synaptic

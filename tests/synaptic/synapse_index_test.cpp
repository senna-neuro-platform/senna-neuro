#include "core/synaptic/synapse_index.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>

namespace senna::synaptic {
namespace {

class SynapseIndexTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSeed = 42;
  static constexpr double kExcRatio = 0.8;

  // Small lattice for unit tests.
  spatial::Lattice lattice_{5, 5, 3, 1.0, kSeed};
  spatial::NeighborIndex neighbors_{lattice_, 2.0f, 1};
  neural::NeuronPool pool_{lattice_, neural::kDefaultLIF, kExcRatio, kSeed};
  SynapseIndex synapses_{lattice_, neighbors_, pool_};
};

// --- 3.1 Synapse structure ---

TEST_F(SynapseIndexTest, SynapseCountPositive) {
  EXPECT_GT(synapses_.synapse_count(), 0);
}

TEST_F(SynapseIndexTest, AllSynapsesHaveValidIds) {
  int n = pool_.size();
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    EXPECT_GE(s.pre_id, 0);
    EXPECT_LT(s.pre_id, n);
    EXPECT_GE(s.post_id, 0);
    EXPECT_LT(s.post_id, n);
    EXPECT_NE(s.pre_id, s.post_id);
  }
}

TEST_F(SynapseIndexTest, WeightsInRange) {
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    EXPECT_GE(s.weight, 0.01f);
    EXPECT_LE(s.weight, 0.1f);
  }
}

TEST_F(SynapseIndexTest, DelayMatchesDistance) {
  // delay = distance * c_base (1.0), so delay == distance for default params.
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    EXPECT_GT(s.delay, 0.0f);
    // Distance between neighbors is at most radius (2.0).
    EXPECT_LE(s.delay, 2.0f + 1e-5f);
  }
}

// --- 3.2 Sign matches presynaptic type ---

TEST_F(SynapseIndexTest, SignMatchesPreType) {
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    float expected_sign = pool_.sign(s.pre_id);
    EXPECT_FLOAT_EQ(s.sign, expected_sign);
  }
}

TEST_F(SynapseIndexTest, EffectiveSign) {
  // Excitatory pre -> positive effective, inhibitory -> negative.
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    if (pool_.type(s.pre_id) == neural::NeuronType::Excitatory) {
      EXPECT_GT(s.Effective(), 0.0f);
    } else {
      EXPECT_LT(s.Effective(), 0.0f);
    }
  }
}

// --- 3.3 CSR index correctness ---

TEST_F(SynapseIndexTest, IncomingIndexCorrect) {
  // Every synapse with post_id == X should appear in Incoming(X).
  int n = pool_.size();
  for (int post = 0; post < n; ++post) {
    auto incoming = synapses_.Incoming(post);
    for (auto sid : incoming) {
      EXPECT_EQ(synapses_.Get(sid).post_id, post);
    }
  }
}

TEST_F(SynapseIndexTest, OutgoingIndexCorrect) {
  int n = pool_.size();
  for (int pre = 0; pre < n; ++pre) {
    auto outgoing = synapses_.Outgoing(pre);
    for (auto sid : outgoing) {
      EXPECT_EQ(synapses_.Get(sid).pre_id, pre);
    }
  }
}

TEST_F(SynapseIndexTest, IncomingOutgoingCoverAll) {
  // Total incoming entries == total outgoing entries == synapse_count.
  int n = pool_.size();
  int total_in = 0, total_out = 0;
  for (int i = 0; i < n; ++i) {
    total_in += synapses_.IncomingCount(i);
    total_out += synapses_.OutgoingCount(i);
  }
  EXPECT_EQ(total_in, synapses_.synapse_count());
  EXPECT_EQ(total_out, synapses_.synapse_count());
}

TEST_F(SynapseIndexTest, SymmetricConnections) {
  // If synapse (A->B) exists, then (B->A) should also exist (from neighbor
  // symmetry).
  std::set<std::pair<int, int>> pairs;
  for (int i = 0; i < synapses_.synapse_count(); ++i) {
    const auto& s = synapses_.Get(i);
    pairs.emplace(s.pre_id, s.post_id);
  }
  for (const auto& [a, b] : pairs) {
    EXPECT_TRUE(pairs.count({b, a}))
        << "Missing reverse synapse: " << b << " -> " << a;
  }
}

// --- 3.4 WTA ---

class WtaSynapseTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSeed = 42;
  static constexpr int kNumOutputs = 4;

  spatial::Lattice lattice_{5, 5, 3, 1.0, kSeed};
  spatial::NeighborIndex neighbors_{lattice_, 2.0f, 1};
  neural::NeuronPool pool_{lattice_, neural::kDefaultLIF, 0.8, kSeed};

  // Fake output neuron IDs (first kNumOutputs neurons).
  std::vector<int32_t> output_ids_{0, 1, 2, 3};
  SynapseIndex synapses_{lattice_, neighbors_, pool_, output_ids_};
};

TEST_F(WtaSynapseTest, WtaCount) {
  // N outputs -> N*(N-1) WTA synapses.
  EXPECT_EQ(synapses_.wta_count(), kNumOutputs * (kNumOutputs - 1));
}

TEST_F(WtaSynapseTest, WtaSynapsesAreInhibitory) {
  // WTA synapses are the last wta_count_ entries.
  int total = synapses_.synapse_count();
  int wta_start = total - synapses_.wta_count();
  for (int i = wta_start; i < total; ++i) {
    const auto& s = synapses_.Get(i);
    EXPECT_FLOAT_EQ(s.sign, -1.0f);
    EXPECT_FLOAT_EQ(s.delay, 0.0f);
    EXPECT_FLOAT_EQ(s.weight, 5.0f);  // |w_wta|
    EXPECT_LT(s.Effective(), 0.0f);
  }
}

TEST_F(WtaSynapseTest, WtaFullyConnected) {
  // Each output neuron should have (kNumOutputs-1) WTA outgoing.
  int total = synapses_.synapse_count();
  int wta_start = total - synapses_.wta_count();

  for (int out_id : output_ids_) {
    int wta_out = 0;
    for (auto sid : synapses_.Outgoing(out_id)) {
      const auto& s = synapses_.Get(sid);
      if (s.sign == -1.0f && s.delay == 0.0f && s.weight == 5.0f) {
        ++wta_out;
      }
    }
    EXPECT_EQ(wta_out, kNumOutputs - 1);
  }
}

TEST_F(WtaSynapseTest, WtaTargetsAllOthers) {
  // When output 0 fires, all other outputs receive inhibition.
  std::set<int> targets;
  for (auto sid : synapses_.Outgoing(0)) {
    const auto& s = synapses_.Get(sid);
    if (s.sign == -1.0f && s.delay == 0.0f && s.weight == 5.0f) {
      targets.insert(s.post_id);
    }
  }
  for (int i = 1; i < kNumOutputs; ++i) {
    EXPECT_TRUE(targets.count(output_ids_[i]))
        << "Output 0 does not inhibit output " << i;
  }
}

}  // namespace
}  // namespace senna::synaptic

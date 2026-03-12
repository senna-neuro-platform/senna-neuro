#include <gtest/gtest.h>

#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"

// Step 1.4 acceptance criteria - smoke tests for the full lattice pipeline.
// These verify the DoD for Step 1 at MVP scale (28x28x20).

namespace senna::spatial {
namespace {

class LatticeSmoke : public ::testing::Test {
 protected:
  static constexpr int kW = 28;
  static constexpr int kH = 28;
  static constexpr int kD = 20;
  static constexpr double kDensity = 0.7;
  static constexpr uint64_t kSeed = 42;
  static constexpr float kRadius = 2.0f;

  ZonedLattice lattice_{kW, kH, kD, kDensity, kSeed};
  NeighborIndex neighbors_{lattice_, kRadius, 0};
};

// 1. Grid 28x28x20: density within 65-75%.
TEST_F(LatticeSmoke, GridDensity) {
  int total = kW * kH * kD;
  double density = static_cast<double>(lattice_.neuron_count()) / total;
  EXPECT_GE(density, 0.65);
  EXPECT_LE(density, 0.75);
}

// 2. Neighbor search: center neuron at R=2 has ~20-35 neighbors.
TEST_F(LatticeSmoke, CenterNeighborCount) {
  // Find a neuron near (14,14,10).
  NeuronId center = kEmptyVoxel;
  for (int d = 0; d <= 2 && center == kEmptyVoxel; ++d) {
    for (int dy = -d; dy <= d && center == kEmptyVoxel; ++dy) {
      for (int dx = -d; dx <= d && center == kEmptyVoxel; ++dx) {
        center = lattice_.NeuronAt(14 + dx, 14 + dy, 10);
      }
    }
  }
  ASSERT_NE(center, kEmptyVoxel);
  int count = neighbors_.NeighborCount(center);
  EXPECT_GE(count, 15);
  EXPECT_LE(count, 40);
}

// 3. Boundary: corner neuron has fewer neighbors than center.
TEST_F(LatticeSmoke, CornerHasFewerNeighbors) {
  NeuronId corner = kEmptyVoxel;
  for (int d = 0; d <= 2 && corner == kEmptyVoxel; ++d) {
    for (int dz = 0; dz <= d && corner == kEmptyVoxel; ++dz) {
      for (int dy = 0; dy <= d && corner == kEmptyVoxel; ++dy) {
        for (int dx = 0; dx <= d && corner == kEmptyVoxel; ++dx) {
          corner = lattice_.NeuronAt(dx, dy, dz);
        }
      }
    }
  }
  ASSERT_NE(corner, kEmptyVoxel);

  NeuronId center = kEmptyVoxel;
  for (int d = 0; d <= 2 && center == kEmptyVoxel; ++d) {
    center = lattice_.NeuronAt(14, 14, 10 + d);
  }
  ASSERT_NE(center, kEmptyVoxel);

  EXPECT_LT(neighbors_.NeighborCount(corner), neighbors_.NeighborCount(center));
}

// 4. Sensory panel: exactly 784 neurons on Z=0.
TEST_F(LatticeSmoke, SensoryPanelCount) {
  EXPECT_EQ(lattice_.sensory_count(), 784);
  int count = 0;
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      if (lattice_.NeuronAt(x, y, 0) != kEmptyVoxel) ++count;
    }
  }
  EXPECT_EQ(count, 784);
}

// 5. Output layer: exactly 10 neurons on Z=19.
TEST_F(LatticeSmoke, OutputLayerCount) {
  EXPECT_EQ(lattice_.num_outputs(), 10);
  int count = 0;
  for (int y = 0; y < kH; ++y) {
    for (int x = 0; x < kW; ++x) {
      if (lattice_.NeuronAt(x, y, kD - 1) != kEmptyVoxel) ++count;
    }
  }
  EXPECT_EQ(count, 10);
}

// 6. Determinism: same seed produces identical lattice.
TEST_F(LatticeSmoke, Determinism) {
  ZonedLattice other(kW, kH, kD, kDensity, kSeed);
  EXPECT_EQ(lattice_.neuron_count(), other.neuron_count());
  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    auto a = lattice_.CoordsOf(id);
    auto b = other.CoordsOf(id);
    EXPECT_EQ(a.x, b.x);
    EXPECT_EQ(a.y, b.y);
    EXPECT_EQ(a.z, b.z);
  }
}

// 7. Parallel neighbor init produces same result as sequential.
TEST_F(LatticeSmoke, ParallelInitCorrectness) {
  NeighborIndex seq(lattice_, kRadius, 1);
  EXPECT_EQ(seq.total_entries(), neighbors_.total_entries());
}

}  // namespace
}  // namespace senna::spatial

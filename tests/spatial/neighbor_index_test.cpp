#include "core/spatial/neighbor_index.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <set>

namespace senna::spatial {
namespace {

// Small lattice for fast tests.
class NeighborIndexTest : public ::testing::Test {
 protected:
  static constexpr int kW = 10;
  static constexpr int kH = 10;
  static constexpr int kD = 10;
  static constexpr double kDensity = 0.7;
  static constexpr uint64_t kSeed = 42;

  Lattice lattice_{kW, kH, kD, kDensity, kSeed};
};

TEST_F(NeighborIndexTest, AllNeighborsWithinRadius) {
  constexpr float R = 2.0F;
  NeighborIndex idx(lattice_, R, 1);

  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    for (auto [nid, dist] : idx.Neighbors(id)) {
      EXPECT_LE(dist, R + 1e-5F);
      EXPECT_GT(dist, 0.0F);
      EXPECT_NE(nid, id);
    }
  }
}

TEST_F(NeighborIndexTest, NeighborsAreSymmetric) {
  constexpr float R = 2.0F;
  NeighborIndex idx(lattice_, R, 1);

  // If B is a neighbor of A, then A must be a neighbor of B.
  for (NeuronId a = 0; a < lattice_.neuron_count(); ++a) {
    for (auto [b, dist_ab] : idx.Neighbors(a)) {
      bool found = false;
      for (auto [c, dist_ba] : idx.Neighbors(b)) {
        if (c == a) {
          found = true;
          EXPECT_NEAR(dist_ab, dist_ba, 1e-5F);
          break;
        }
      }
      EXPECT_TRUE(found) << "Neuron " << b << " is neighbor of " << a
                         << " but not vice versa";
    }
  }
}

TEST_F(NeighborIndexTest, CenterNeuronHasMoreNeighborsThanCorner) {
  constexpr float R = 2.0F;
  NeighborIndex idx(lattice_, R, 1);

  // Find a neuron near center (5,5,5).
  NeuronId center_id = kEmptyVoxel;
  for (int dz = 0; dz <= 2 && center_id == kEmptyVoxel; ++dz) {
    for (int dy = 0; dy <= 2 && center_id == kEmptyVoxel; ++dy) {
      for (int dx = 0; dx <= 2 && center_id == kEmptyVoxel; ++dx) {
        center_id = lattice_.NeuronAt(5 + dx, 5 + dy, 5 + dz);
      }
    }
  }
  ASSERT_NE(center_id, kEmptyVoxel);

  // Find a neuron at corner (0,0,0).
  NeuronId corner_id = kEmptyVoxel;
  for (int dz = 0; dz <= 2 && corner_id == kEmptyVoxel; ++dz) {
    for (int dy = 0; dy <= 2 && corner_id == kEmptyVoxel; ++dy) {
      for (int dx = 0; dx <= 2 && corner_id == kEmptyVoxel; ++dx) {
        corner_id = lattice_.NeuronAt(dx, dy, dz);
      }
    }
  }
  ASSERT_NE(corner_id, kEmptyVoxel);

  EXPECT_GT(idx.NeighborCount(center_id), idx.NeighborCount(corner_id));
}

TEST_F(NeighborIndexTest, ExpectedNeighborCountForR2) {
  constexpr float R = 2.0F;
  NeighborIndex idx(lattice_, R, 1);

  // For a neuron in the center: sphere of radius 2 has volume 4/3*pi*8 ~ 33.5
  // At density 0.7: ~23 neighbors (minus self). Allow wide range due to
  // discrete grid and boundary effects.
  NeuronId center_id = kEmptyVoxel;
  for (int dz = 0; dz <= 2 && center_id == kEmptyVoxel; ++dz) {
    for (int dy = 0; dy <= 2 && center_id == kEmptyVoxel; ++dy) {
      for (int dx = 0; dx <= 2 && center_id == kEmptyVoxel; ++dx) {
        center_id = lattice_.NeuronAt(5 + dx, 5 + dy, 5 + dz);
      }
    }
  }
  ASSERT_NE(center_id, kEmptyVoxel);

  int count = idx.NeighborCount(center_id);
  // Discrete sphere of R=2: voxels at distance <= 2.0 are about 33
  // (cube 5x5x5=125, minus those outside sphere). At density 0.7 ~ 15-25.
  EXPECT_GE(count, 10);
  EXPECT_LE(count, 35);
}

TEST_F(NeighborIndexTest, NoSelfInNeighbors) {
  constexpr float R = 3.0F;
  NeighborIndex idx(lattice_, R, 1);

  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    for (auto [nid, dist] : idx.Neighbors(id)) {
      EXPECT_NE(nid, id);
    }
  }
}

TEST_F(NeighborIndexTest, DistancesAreCorrect) {
  constexpr float R = 2.0F;
  NeighborIndex idx(lattice_, R, 1);

  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    auto [cx, cy, cz] = lattice_.CoordsOf(id);
    for (auto [nid, dist] : idx.Neighbors(id)) {
      auto [nx, ny, nz] = lattice_.CoordsOf(nid);
      float dx = static_cast<float>(nx - cx);
      float dy = static_cast<float>(ny - cy);
      float dz = static_cast<float>(nz - cz);
      float expected = std::sqrt(dx * dx + dy * dy + dz * dz);
      EXPECT_NEAR(dist, expected, 1e-5F);
    }
  }
}

TEST_F(NeighborIndexTest, ParallelAndSequentialGiveSameResult) {
  constexpr float R = 2.0F;
  NeighborIndex seq(lattice_, R, 1);
  NeighborIndex par(lattice_, R, 4);

  EXPECT_EQ(seq.total_entries(), par.total_entries());

  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    auto sn = seq.Neighbors(id);
    auto pn = par.Neighbors(id);
    ASSERT_EQ(sn.size(), pn.size()) << "Mismatch for neuron " << id;

    // Build sets for comparison (order may differ between threads).
    std::set<NeuronId> s_ids, p_ids;
    for (auto& e : sn) s_ids.insert(e.id);
    for (auto& e : pn) p_ids.insert(e.id);
    EXPECT_EQ(s_ids, p_ids) << "Neighbor set mismatch for neuron " << id;
  }
}

TEST_F(NeighborIndexTest, ZeroRadiusGivesNoNeighbors) {
  NeighborIndex idx(lattice_, 0.0F, 1);
  for (NeuronId id = 0; id < lattice_.neuron_count(); ++id) {
    EXPECT_EQ(idx.NeighborCount(id), 0);
  }
}

TEST_F(NeighborIndexTest, EmptyLatticeWorks) {
  Lattice empty(5, 5, 5, 0.0, 42);
  EXPECT_EQ(empty.neuron_count(), 0);
  NeighborIndex idx(empty, 2.0F, 1);
  EXPECT_EQ(idx.total_entries(), 0u);
}

TEST_F(NeighborIndexTest, FullDensityLattice) {
  Lattice full(5, 5, 5, 1.0, 42);
  EXPECT_EQ(full.neuron_count(), 125);
  NeighborIndex idx(full, 1.0F, 2);

  // Center neuron (2,2,2) should have 6 neighbors at distance 1.0 (face
  // neighbors).
  NeuronId center = full.NeuronAt(2, 2, 2);
  ASSERT_NE(center, kEmptyVoxel);
  EXPECT_EQ(idx.NeighborCount(center), 6);

  // Corner neuron (0,0,0) should have 3 neighbors at distance 1.0.
  NeuronId corner = full.NeuronAt(0, 0, 0);
  ASSERT_NE(corner, kEmptyVoxel);
  EXPECT_EQ(idx.NeighborCount(corner), 3);
}

}  // namespace
}  // namespace senna::spatial

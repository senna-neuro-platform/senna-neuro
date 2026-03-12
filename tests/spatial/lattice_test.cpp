#include "core/spatial/lattice.hpp"

#include <gtest/gtest.h>

namespace senna::spatial {
namespace {

TEST(LatticeTest, DimensionsAreCorrect) {
  Lattice lat(28, 28, 20, 0.7, 42);
  EXPECT_EQ(lat.width(), 28);
  EXPECT_EQ(lat.height(), 28);
  EXPECT_EQ(lat.depth(), 20);
}

TEST(LatticeTest, DensityInRange) {
  Lattice lat(28, 28, 20, 0.7, 42);
  int total_voxels = 28 * 28 * 20;
  double density =
      static_cast<double>(lat.neuron_count()) / total_voxels;
  EXPECT_GE(density, 0.65);
  EXPECT_LE(density, 0.75);
}

TEST(LatticeTest, NeuronAtReturnsValidIds) {
  Lattice lat(10, 10, 10, 0.5, 123);
  int occupied = 0;
  for (int z = 0; z < 10; ++z) {
    for (int y = 0; y < 10; ++y) {
      for (int x = 0; x < 10; ++x) {
        NeuronId id = lat.NeuronAt(x, y, z);
        if (id != kEmptyVoxel) {
          ++occupied;
          EXPECT_GE(id, 0);
          EXPECT_LT(id, lat.neuron_count());
        }
      }
    }
  }
  EXPECT_EQ(occupied, lat.neuron_count());
}

TEST(LatticeTest, CoordsOfRoundTrips) {
  Lattice lat(28, 28, 20, 0.7, 42);
  for (NeuronId id = 0; id < lat.neuron_count(); ++id) {
    auto [x, y, z] = lat.CoordsOf(id);
    EXPECT_EQ(lat.NeuronAt(x, y, z), id);
  }
}

TEST(LatticeTest, DeterministicWithSameSeed) {
  Lattice a(28, 28, 20, 0.7, 42);
  Lattice b(28, 28, 20, 0.7, 42);
  EXPECT_EQ(a.neuron_count(), b.neuron_count());
  for (NeuronId id = 0; id < a.neuron_count(); ++id) {
    auto ca = a.CoordsOf(id);
    auto cb = b.CoordsOf(id);
    EXPECT_EQ(ca.x, cb.x);
    EXPECT_EQ(ca.y, cb.y);
    EXPECT_EQ(ca.z, cb.z);
  }
}

TEST(LatticeTest, DifferentSeedGivesDifferentLattice) {
  Lattice a(28, 28, 20, 0.7, 42);
  Lattice b(28, 28, 20, 0.7, 99);
  // Very unlikely to have the same count AND layout.
  bool differ = (a.neuron_count() != b.neuron_count());
  if (!differ) {
    for (NeuronId id = 0; id < a.neuron_count(); ++id) {
      auto ca = a.CoordsOf(id);
      auto cb = b.CoordsOf(id);
      if (ca.x != cb.x || ca.y != cb.y || ca.z != cb.z) {
        differ = true;
        break;
      }
    }
  }
  EXPECT_TRUE(differ);
}

}  // namespace
}  // namespace senna::spatial

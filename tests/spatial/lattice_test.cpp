#include "core/spatial/lattice.hpp"

#include <gtest/gtest.h>

#include <set>
#include <vector>

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
  double density = static_cast<double>(lat.neuron_count()) / total_voxels;
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

// --- ZonedLattice tests ---

TEST(ZonedLatticeTest, SensoryPanelIsFull) {
  ZonedLattice lat(28, 28, 20, 0.7, 42);
  int count = 0;
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      NeuronId id = lat.NeuronAt(x, y, 0);
      EXPECT_NE(id, kEmptyVoxel) << "Empty at (" << x << "," << y << ",0)";
      if (id != kEmptyVoxel) ++count;
    }
  }
  EXPECT_EQ(count, 784);  // 28 * 28
  EXPECT_EQ(lat.sensory_count(), 784);
}

TEST(ZonedLatticeTest, SensoryNeuronAccessor) {
  ZonedLattice lat(28, 28, 20, 0.7, 42);
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      EXPECT_EQ(lat.SensoryNeuron(x, y), lat.NeuronAt(x, y, 0));
    }
  }
}

TEST(ZonedLatticeTest, OutputLayerHasExactly10Neurons) {
  ZonedLattice lat(28, 28, 20, 0.7, 42, 10);
  EXPECT_EQ(lat.num_outputs(), 10);

  // Count occupied voxels on Z=19.
  int count = 0;
  for (int y = 0; y < 28; ++y) {
    for (int x = 0; x < 28; ++x) {
      if (lat.NeuronAt(x, y, 19) != kEmptyVoxel) ++count;
    }
  }
  EXPECT_EQ(count, 10);
}

TEST(ZonedLatticeTest, OutputNeuronIdsAreValid) {
  ZonedLattice lat(28, 28, 20, 0.7, 42, 10);
  for (int i = 0; i < 10; ++i) {
    NeuronId id = lat.OutputNeuron(i);
    EXPECT_GE(id, 0);
    EXPECT_LT(id, lat.neuron_count());
    auto [x, y, z] = lat.CoordsOf(id);
    EXPECT_EQ(z, 19) << "Output neuron " << i << " not on Z=19";
  }
}

TEST(ZonedLatticeTest, OutputNeuronsAreDistinct) {
  ZonedLattice lat(28, 28, 20, 0.7, 42, 10);
  std::set<NeuronId> ids;
  for (int i = 0; i < 10; ++i) {
    ids.insert(lat.OutputNeuron(i));
  }
  EXPECT_EQ(ids.size(), 10u);
}

TEST(ZonedLatticeTest, OutputNeuronsEvenlySpaced) {
  ZonedLattice lat(28, 28, 20, 0.7, 42, 10);
  std::vector<int> xs;
  for (int i = 0; i < 10; ++i) {
    auto [x, y, z] = lat.CoordsOf(lat.OutputNeuron(i));
    xs.push_back(x);
  }
  // Should be monotonically increasing (evenly spread along X).
  for (int i = 1; i < 10; ++i) {
    EXPECT_GT(xs[i], xs[i - 1]);
  }
}

TEST(ZonedLatticeTest, ProcessingVolumeHasStandardDensity) {
  ZonedLattice lat(28, 28, 20, 0.7, 42);
  int occupied = 0;
  int total = 0;
  for (int z = 1; z <= 18; ++z) {
    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        ++total;
        if (lat.NeuronAt(x, y, z) != kEmptyVoxel) ++occupied;
      }
    }
  }
  double density = static_cast<double>(occupied) / total;
  EXPECT_GE(density, 0.65);
  EXPECT_LE(density, 0.75);
}

TEST(ZonedLatticeTest, CoordsOfRoundTrips) {
  ZonedLattice lat(28, 28, 20, 0.7, 42);
  // ClearVoxel leaves "orphan" neuron IDs - skip those.
  for (int z = 0; z < 20; ++z) {
    for (int y = 0; y < 28; ++y) {
      for (int x = 0; x < 28; ++x) {
        NeuronId id = lat.NeuronAt(x, y, z);
        if (id != kEmptyVoxel) {
          auto c = lat.CoordsOf(id);
          EXPECT_EQ(c.x, x);
          EXPECT_EQ(c.y, y);
          EXPECT_EQ(c.z, z);
        }
      }
    }
  }
}

}  // namespace
}  // namespace senna::spatial

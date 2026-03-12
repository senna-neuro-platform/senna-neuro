#include "core/spatial/lattice.hpp"

#include <random>

namespace senna::spatial {

Lattice::Lattice(int width, int height, int depth, double density,
                 uint64_t seed)
    : width_(width),
      height_(height),
      depth_(depth),
      grid_(width * height * depth, kEmptyVoxel) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (int z = 0; z < depth_; ++z) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        if (dist(rng) < density) {
          PlaceNeuron(x, y, z);
        }
      }
    }
  }
}

NeuronId Lattice::NeuronAt(int x, int y, int z) const {
  return grid_[VoxelIndex(x, y, z)];
}

NeuronCoords Lattice::CoordsOf(NeuronId id) const { return neurons_[id]; }

void Lattice::PlaceNeuron(int x, int y, int z) {
  NeuronId id = static_cast<NeuronId>(neurons_.size());
  neurons_.push_back({x, y, z});
  grid_[VoxelIndex(x, y, z)] = id;
}

void Lattice::ClearVoxel(int x, int y, int z) {
  grid_[VoxelIndex(x, y, z)] = kEmptyVoxel;
}

int Lattice::VoxelIndex(int x, int y, int z) const {
  return z * width_ * height_ + y * width_ + x;
}

// --- ZonedLattice ---

ZonedLattice::ZonedLattice(int width, int height, int depth, double density,
                           uint64_t seed, int num_outputs)
    : Lattice(width, height, depth, density, seed), num_outputs_(num_outputs) {
  // --- Sensory panel: Z=0 must be 100% filled ---
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (NeuronAt(x, y, 0) == kEmptyVoxel) {
        PlaceNeuron(x, y, 0);
      }
    }
  }

  // --- Output layer: Z=depth-1, exactly num_outputs neurons ---
  int out_z = depth - 1;

  // First, clear all existing neurons on the output plane.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (NeuronAt(x, y, out_z) != kEmptyVoxel) {
        ClearVoxel(x, y, out_z);
      }
    }
  }

  // Place num_outputs neurons evenly spaced across the output plane.
  // Distribute along a grid: spread in X/Y dimensions.
  output_ids_.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    // Spread evenly along a line in X at the center Y.
    int x = static_cast<int>((i + 0.5) * width / num_outputs);
    int y = height / 2;
    PlaceNeuron(x, y, out_z);
    output_ids_.push_back(NeuronAt(x, y, out_z));
  }
}

NeuronId ZonedLattice::SensoryNeuron(int x, int y) const {
  return NeuronAt(x, y, 0);
}

}  // namespace senna::spatial

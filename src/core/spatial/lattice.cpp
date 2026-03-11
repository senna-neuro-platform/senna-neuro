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

}  // namespace senna::spatial

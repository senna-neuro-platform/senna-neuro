#pragma once

#include <cstdint>
#include <vector>

namespace senna::spatial {

using NeuronId = int32_t;
inline constexpr NeuronId kEmptyVoxel = -1;

// Coordinates of a neuron in the 3D lattice.
struct NeuronCoords {
  int x;
  int y;
  int z;
};

// 3D lattice of voxels. Each voxel is either empty or holds a neuron ID.
// Neurons are stored in a flat vector; voxels store indices into that vector.
// Construction is deterministic given the same seed.
class Lattice {
 public:
  // Fills the lattice with neurons at the given density (0.0–1.0).
  // Voxel (x, y, z) is visited in Z-major, Y-minor, X-inner order.
  Lattice(int width, int height, int depth, double density, uint64_t seed);

  // Returns the neuron ID at voxel (x,y,z), or kEmptyVoxel if empty.
  NeuronId NeuronAt(int x, int y, int z) const;

  // Returns the lattice coordinates of a neuron by its ID.
  NeuronCoords CoordsOf(NeuronId id) const;

  int width() const { return width_; }
  int height() const { return height_; }
  int depth() const { return depth_; }
  int neuron_count() const { return static_cast<int>(neurons_.size()); }

 protected:
  // Allows subclasses (e.g. for zone overrides) to place a neuron directly.
  void PlaceNeuron(int x, int y, int z);

  // Removes a neuron from a voxel (marks it kEmptyVoxel).
  // Does not shrink the neuron vector — ID remains valid.
  void ClearVoxel(int x, int y, int z);

  int VoxelIndex(int x, int y, int z) const;

 private:
  int width_;
  int height_;
  int depth_;
  std::vector<NeuronCoords> neurons_;  // indexed by NeuronId
  std::vector<NeuronId> grid_;         // W*H*D, kEmptyVoxel if empty
};

}  // namespace senna::spatial

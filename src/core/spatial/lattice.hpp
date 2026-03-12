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
  // Does not shrink the neuron vector - ID remains valid.
  void ClearVoxel(int x, int y, int z);

  int VoxelIndex(int x, int y, int z) const;

 private:
  int width_;
  int height_;
  int depth_;
  std::vector<NeuronCoords> neurons_;  // indexed by NeuronId
  std::vector<NeuronId> grid_;         // W*H*D, kEmptyVoxel if empty
};

// Lattice with enforced zones for SENNA MVP:
//   - Sensory panel: Z=0, 100% density (W*H neurons)
//   - Processing volume: Z=1..D-2, standard density
//   - Output layer: Z=D-1, exactly num_outputs neurons evenly spaced
class ZonedLattice : public Lattice {
 public:
  // num_outputs: number of output neurons on Z=D-1 (10 for MNIST).
  ZonedLattice(int width, int height, int depth, double density, uint64_t seed,
               int num_outputs = 10);

  int num_outputs() const { return num_outputs_; }

  // Returns the NeuronId of the i-th output neuron (0-based).
  NeuronId OutputNeuron(int index) const { return output_ids_[index]; }

  // Returns the NeuronId of the sensory neuron at (x, y) on Z=0.
  NeuronId SensoryNeuron(int x, int y) const;

  // Number of sensory neurons (always width * height).
  int sensory_count() const { return width() * height(); }

 private:
  int num_outputs_;
  std::vector<NeuronId> output_ids_;
};

}  // namespace senna::spatial

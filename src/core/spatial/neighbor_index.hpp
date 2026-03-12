#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "core/spatial/lattice.hpp"

namespace senna::spatial {

// A neighbor entry: neuron ID and Euclidean distance to it.
struct NeighborEntry {
  NeuronId id;
  float distance;
};

// Pre-computed neighbor lists for every neuron in the lattice.
// Stored in CSR (Compressed Sparse Row) format for cache efficiency.
//
// For neuron i, its neighbors are:
//   data_[offsets_[i] .. offsets_[i+1])
//
// Thread-safe for concurrent reads after construction.
class NeighborIndex {
 public:
  // Builds the neighbor index for all neurons in the lattice.
  // radius: maximum Euclidean distance for a neighbor.
  // num_threads: number of threads for parallel construction (0 = hardware
  // concurrency).
  NeighborIndex(const Lattice& lattice, float radius, unsigned num_threads = 0);

  // Returns a span of neighbors for the given neuron.
  std::span<const NeighborEntry> Neighbors(NeuronId id) const;

  // Number of neighbors for a given neuron.
  int NeighborCount(NeuronId id) const;

  // Total number of neighbor entries across all neurons.
  size_t total_entries() const { return data_.size(); }

  float radius() const { return radius_; }

 private:
  float radius_;
  std::vector<uint32_t> offsets_;    // size = neuron_count + 1
  std::vector<NeighborEntry> data_;  // flat array of all neighbor entries
};

}  // namespace senna::spatial

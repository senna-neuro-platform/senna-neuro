#include "core/spatial/neighbor_index.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace senna::spatial {

NeighborIndex::NeighborIndex(const Lattice& lattice, float radius,
                             unsigned num_threads)
    : radius_(radius) {
  const int n = lattice.neuron_count();
  if (n == 0) {
    offsets_.assign(1, 0);
    return;
  }

  const int R = static_cast<int>(std::ceil(radius));
  const float r_sq = radius * radius;
  const int W = lattice.width();
  const int H = lattice.height();
  const int D = lattice.depth();

  // Phase 1: each thread builds a local neighbor list per neuron.
  // We partition neurons across threads.
  if (num_threads == 0) {
    num_threads = std::max(1U, std::thread::hardware_concurrency());
  }
  num_threads = std::min(num_threads, static_cast<unsigned>(n));

  // Per-neuron local storage: vector of NeighborEntry per neuron.
  std::vector<std::vector<NeighborEntry>> per_neuron(n);

  auto worker = [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      auto [cx, cy, cz] = lattice.CoordsOf(i);

      int x_lo = std::max(0, cx - R);
      int x_hi = std::min(W - 1, cx + R);
      int y_lo = std::max(0, cy - R);
      int y_hi = std::min(H - 1, cy + R);
      int z_lo = std::max(0, cz - R);
      int z_hi = std::min(D - 1, cz + R);

      for (int z = z_lo; z <= z_hi; ++z) {
        for (int y = y_lo; y <= y_hi; ++y) {
          for (int x = x_lo; x <= x_hi; ++x) {
            if (x == cx && y == cy && z == cz) {
              continue;
            }

            NeuronId nid = lattice.NeuronAt(x, y, z);
            if (nid == kEmptyVoxel) {
              continue;
            }

            auto dx = static_cast<float>(x - cx);
            auto dy = static_cast<float>(y - cy);
            auto dz = static_cast<float>(z - cz);
            float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq <= r_sq) {
              per_neuron[i].push_back({nid, std::sqrt(dist_sq)});
            }
          }
        }
      }
    }
  };

  // Launch threads.
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  int chunk = n / static_cast<int>(num_threads);
  int remainder = n % static_cast<int>(num_threads);
  int start = 0;

  for (unsigned t = 0; t < num_threads; ++t) {
    int end = start + chunk + (static_cast<int>(t) < remainder ? 1 : 0);
    threads.emplace_back(worker, start, end);
    start = end;
  }

  for (auto& th : threads) {
    th.join();
  }

  // Phase 2: compact into CSR format.
  offsets_.resize(n + 1);
  offsets_[0] = 0;
  for (int i = 0; i < n; ++i) {
    offsets_[i + 1] = offsets_[i] + static_cast<uint32_t>(per_neuron[i].size());
  }

  data_.resize(offsets_[n]);
  for (int i = 0; i < n; ++i) {
    std::copy(per_neuron[i].begin(), per_neuron[i].end(),
              data_.begin() + offsets_[i]);
  }
}

std::span<const NeighborEntry> NeighborIndex::Neighbors(NeuronId id) const {
  return {data_.data() + offsets_[id], data_.data() + offsets_[id + 1]};
}

int NeighborIndex::NeighborCount(NeuronId id) const {
  return static_cast<int>(offsets_[id + 1] - offsets_[id]);
}

}  // namespace senna::spatial

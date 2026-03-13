#include "core/plasticity/stdp_worker.hpp"

#include <gtest/gtest.h>

#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"

namespace senna::plasticity {
namespace {

struct WorkerNet {
  spatial::Lattice lattice{2, 1, 1, 1.0, 7};
  spatial::NeighborIndex neighbors{lattice, 2.0F, 1};
  neural::NeuronPool pool{lattice, neural::kDefaultLIF, 1.0, 99};
  std::shared_ptr<synaptic::SynapseIndex> syn_index =
      std::make_shared<synaptic::SynapseIndex>(lattice, neighbors, pool);
  std::atomic<std::shared_ptr<synaptic::SynapseIndex>> syn_store{syn_index};
  synaptic::SynapseParams syn_params{};
  STDPParams params{};
};

TEST(STDPWorkerTest, ProcessesEnqueuedSpikes) {
  WorkerNet net;
  auto sid = net.syn_index->Outgoing(0)[0];
  float w0 = net.syn_index->Get(sid).weight;

  STDPWorker worker(net.syn_store, net.pool, net.syn_params, net.params);
  worker.Start();

  // Pre then post; worker should process both and potentiate.
  net.pool.Fire(0, 0.0F);
  net.pool.Fire(1, 5.0F);
  worker.Enqueue(0, 0.0F);
  worker.Enqueue(1, 5.0F);

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  worker.Stop();

  float w1 = net.syn_index->Get(sid).weight;
  EXPECT_GT(w1, w0);
}

TEST(STDPWorkerTest, StopsCleanly) {
  WorkerNet net;
  STDPWorker worker(net.syn_store, net.pool, net.syn_params, net.params);
  worker.Start();
  worker.Stop();  // should not hang
  SUCCEED();
}

}  // namespace
}  // namespace senna::plasticity

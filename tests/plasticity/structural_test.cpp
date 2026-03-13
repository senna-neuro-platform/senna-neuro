#include "core/plasticity/structural.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "core/spatial/lattice.hpp"
#include "core/spatial/neighbor_index.hpp"

namespace senna::plasticity {

class StructuralTest : public ::testing::Test {
 protected:
  spatial::Lattice lattice_{3, 3, 1, 1.0, 42};  // full grid
  spatial::NeighborIndex neighbors_{lattice_, 1.5F, 0};
  neural::LIFParams lif_{};
};

TEST_F(StructuralTest, PrunesLowWeightSynapses) {
  neural::NeuronPool pool(lattice_, lif_, 1.0, 123);
  synaptic::SynapseParams syn_params;

  // Build simple index.
  synaptic::SynapseIndex idx(lattice_, neighbors_, pool, {}, syn_params, 999);

  // Force one synapse to be tiny.
  synaptic::Synapse& s0 = idx.synapses()[0];
  s0.weight = 0.0001F;

  StructuralConfig cfg;
  cfg.w_min_prune = 0.001F;
  StructuralPlasticity sp(cfg);

  auto pruned = sp.Run(lattice_, neighbors_, pool, idx,
                       /*homeo_target_hz=*/5.0F, syn_params);
  EXPECT_LT(pruned.synapse_count(), idx.synapse_count());
}

TEST_F(StructuralTest, SproutsForQuietNeurons) {
  neural::NeuronPool pool(lattice_, lif_, 1.0, 123);
  synaptic::SynapseParams syn_params;

  synaptic::SynapseIndex idx(lattice_, neighbors_, pool, {}, syn_params, 999);

  // Make neuron 0 very quiet.
  pool.r_avg(0) = 0.0F;
  // Remove all incoming to neuron 0 to ensure sprout is needed.
  std::vector<synaptic::Synapse> filtered;
  for (auto s : idx.synapses()) {
    if (s.post_id != 0) filtered.push_back(s);
  }
  synaptic::SynapseIndex without0(idx.synapse_count(), filtered,
                                  idx.wta_count());

  StructuralConfig cfg;
  cfg.quiet_fraction = 0.9F;
  cfg.sprout_weight = 0.02F;
  StructuralPlasticity sp(cfg);

  auto updated = sp.Run(lattice_, neighbors_, pool, without0,
                        /*homeo_target_hz=*/5.0F, syn_params);

  // Expect at least one incoming synapse to neuron 0 after sprout.
  bool has_incoming0 = false;
  for (const auto& s : updated.synapses()) {
    if (s.post_id == 0) {
      has_incoming0 = true;
      EXPECT_FLOAT_EQ(s.weight, cfg.sprout_weight);
      break;
    }
  }
  EXPECT_TRUE(has_incoming0);
}

TEST_F(StructuralTest, WorkerPrunesInBackground) {
  neural::NeuronPool pool(lattice_, lif_, 1.0, 123);
  synaptic::SynapseParams syn_params;

  auto base_idx = std::make_shared<synaptic::SynapseIndex>(
      lattice_, neighbors_, pool, std::span<const int32_t>{}, syn_params, 999);
  int initial = base_idx->synapse_count();
  // Force a non-WTA synapse to be tiny so it gets pruned.
  for (auto& s : base_idx->synapses()) {
    bool is_wta = (s.delay == 0.0F && s.sign < 0.0F && s.pre_id != s.post_id);
    if (!is_wta) {
      s.weight = 0.00001F;
      break;
    }
  }

  std::atomic<std::shared_ptr<synaptic::SynapseIndex>> store;
  store.store(base_idx);

  StructuralConfig cfg;
  cfg.w_min_prune = 0.001F;
  cfg.interval_ticks = 1;
  cfg.sprout_radius = 0.0F;  // disable sprouting
  cfg.quiet_fraction = 0.0F;

  StructuralWorker worker(lattice_, neighbors_, pool, store, syn_params, cfg,
                          /*homeo_target_hz=*/5.0F);
  worker.Start();
  worker.Trigger();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  worker.Stop();

  auto updated = store.load();
  ASSERT_TRUE(updated);
  EXPECT_LT(updated->synapse_count(), initial);
}

}  // namespace senna::plasticity

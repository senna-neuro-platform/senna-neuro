#include "core/plasticity/structural.hpp"

#include <algorithm>
#include <chrono>
#include <unordered_set>

namespace senna::plasticity {

namespace {

struct PairHash {
  size_t operator()(const std::pair<int, int>& p) const noexcept {
    return (static_cast<size_t>(p.first) << 32) ^ static_cast<size_t>(p.second);
  }
};

}  // namespace

StructuralWorker::StructuralWorker(
    const spatial::Lattice& lattice, const spatial::NeighborIndex& neighbors,
    neural::NeuronPool& pool,
    std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& store,
    synaptic::SynapseParams syn_params, StructuralConfig cfg,
    float homeo_target_hz)
    : lattice_(lattice),
      neighbors_(neighbors),
      pool_(pool),
      store_(store),
      syn_params_(syn_params),
      sp_(cfg),
      cfg_(cfg),
      homeo_target_hz_(homeo_target_hz) {}

StructuralWorker::~StructuralWorker() { Stop(); }

void StructuralWorker::Start() {
  if (running_.exchange(true)) return;
  worker_ = std::thread(&StructuralWorker::Loop, this);
}

void StructuralWorker::Stop() {
  if (!running_.exchange(false)) return;
  {
    std::lock_guard<std::mutex> lk(m_);
    has_task_ = true;  // wake the worker so it can exit
  }
  cv_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void StructuralWorker::Trigger() {
  {
    std::lock_guard<std::mutex> lk(m_);
    has_task_ = true;
  }
  cv_.notify_one();
}

void StructuralWorker::Loop() {
  while (running_.load()) {
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return !running_.load() || has_task_; });
      if (!running_.load()) break;
      has_task_ = false;
    }
    auto current = store_.load();
    if (!current) continue;
    auto updated = std::make_shared<synaptic::SynapseIndex>(sp_.Run(
        lattice_, neighbors_, pool_, *current, homeo_target_hz_, syn_params_));
    store_.store(updated);
  }
}

synaptic::SynapseIndex StructuralPlasticity::Run(
    const spatial::Lattice& lattice, const spatial::NeighborIndex& neighbors,
    const neural::NeuronPool& pool, const synaptic::SynapseIndex& current,
    float homeo_target_hz, const synaptic::SynapseParams& syn_params) const {
  const int n = pool.size();
  const float dt_ms = 1.0f;  // assume r_avg is per-tick; dt cancels in ratio

  // Build adjacency set to test existence quickly.
  std::unordered_set<std::pair<int, int>, PairHash> exists;
  exists.reserve(current.synapse_count() * 2);
  for (const auto& s : current.synapses()) {
    exists.emplace(s.pre_id, s.post_id);
  }

  std::vector<synaptic::Synapse> kept;
  kept.reserve(current.synapse_count());
  int32_t wta_count = current.wta_count();

  // Prune low-weight non-WTA synapses.
  for (const auto& s : current.synapses()) {
    bool is_wta = (s.delay == 0.0f && s.sign < 0.0f && s.pre_id != s.post_id);
    if (!is_wta && std::abs(s.weight) < cfg_.w_min_prune) continue;
    kept.push_back(s);
  }

  // Sprout for quiet neurons: add one new incoming synapse if missing.
  float quiet_thresh_hz = homeo_target_hz * cfg_.quiet_fraction;
  for (int post = 0; post < n; ++post) {
    float freq_hz = pool.r_avg(post) / (dt_ms * 1e-3f);
    if (freq_hz >= quiet_thresh_hz) continue;

    // Search neighbors within sprout_radius.
    for (const auto& nb : neighbors.Neighbors(post)) {
      if (nb.distance > cfg_.sprout_radius) continue;
      int pre = nb.id;
      if (pre == post) continue;
      if (exists.count({pre, post})) continue;

      synaptic::Synapse syn{
          .pre_id = pre,
          .post_id = post,
          .weight = cfg_.sprout_weight,
          .delay = nb.distance * syn_params.c_base,
          .sign = pool.sign(pre),
      };
      kept.push_back(syn);
      exists.emplace(pre, post);
      break;  // add at most one new synapse per quiet neuron
    }
  }

  return synaptic::SynapseIndex(n, std::move(kept), wta_count);
}

}  // namespace senna::plasticity

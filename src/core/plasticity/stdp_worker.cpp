#include "core/plasticity/stdp_worker.hpp"

#include <random>

namespace senna::plasticity {

STDPWorker::STDPWorker(
    std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& syn_store,
    neural::NeuronPool& pool, const synaptic::SynapseParams& syn_params,
    const STDPParams& params, uint64_t /*seed*/)
    : syn_store_(syn_store),
      pool_(pool),
      syn_params_(syn_params),
      params_(params) {}

STDPWorker::~STDPWorker() { Stop(); }

void STDPWorker::Start() {
  if (running_.exchange(true)) {
    return;
  }
  worker_ = std::thread(&STDPWorker::Run, this);
}

void STDPWorker::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  Enqueue(-1, 0.0F);  // wake up
  if (worker_.joinable()) {
    worker_.join();
  }
}

void STDPWorker::Enqueue(int neuron_id, float t_spike) {
  Node* node = new Node{{neuron_id, t_spike}, nullptr};
  Node* old = pending_.load(std::memory_order_relaxed);
  while (true) {
    node->next = old;
    if (pending_.compare_exchange_weak(old, node, std::memory_order_release,
                                       std::memory_order_relaxed)) {
      break;
    }
  }
}

void STDPWorker::DrainPending(std::vector<Spike>& out) {
  Node* list = pending_.exchange(nullptr, std::memory_order_acquire);
  for (Node* n = list; n != nullptr;) {
    out.push_back(n->spike);
    Node* next = n->next;
    delete n;
    n = next;
  }
}

void STDPWorker::Run() {
  std::vector<Spike> batch;
  batch.reserve(256);
  while (true) {
    batch.clear();
    DrainPending(batch);
    if (!running_.load() && batch.empty()) {
      break;
    }
    auto syn_ptr = syn_store_.load();
    if (!syn_ptr) {
      std::this_thread::sleep_for(std::chrono::milliseconds(0));
      continue;
    }
    for (const auto& s : batch) {
      if (s.id < 0) {
        continue;
      }
      STDP::OnPreSpike(s.id, s.t, *syn_ptr, pool_, syn_params_, params_);
      STDP::OnPostSpike(s.id, s.t, *syn_ptr, pool_, syn_params_, params_);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(0));
  }
}

}  // namespace senna::plasticity

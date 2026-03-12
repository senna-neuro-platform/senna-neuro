#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "core/plasticity/stdp.hpp"

namespace senna::plasticity {

// Background STDP worker: consumes spike events and applies pair-based updates.
class STDPWorker {
 public:
  STDPWorker(std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& syn_store,
             neural::NeuronPool& pool,
             const synaptic::SynapseParams& syn_params,
             const STDPParams& params = {}, uint64_t seed = 42);
  ~STDPWorker();

  void Start();
  void Stop();

  // Enqueue a spike (neuron_id, t_spike). MPSC-friendly.
  void Enqueue(int neuron_id, float t_spike);

 private:
  struct Spike {
    int id;
    float t;
  };

  void DrainPending(std::vector<Spike>& out);
  void Run();

  std::atomic<std::shared_ptr<synaptic::SynapseIndex>>& syn_store_;
  neural::NeuronPool& pool_;
  synaptic::SynapseParams syn_params_;
  STDPParams params_;

  std::atomic<bool> running_{false};
  std::thread worker_;

  // lock-free stack for pending spikes
  struct Node {
    Spike spike;
    Node* next;
  };
  std::atomic<Node*> pending_{nullptr};
};

}  // namespace senna::plasticity

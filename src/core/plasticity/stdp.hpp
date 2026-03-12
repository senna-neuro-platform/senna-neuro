#pragma once

#include "core/neural/neuron_pool.hpp"
#include "core/synaptic/synapse.hpp"
#include "core/synaptic/synapse_index.hpp"

namespace senna::plasticity {

struct STDPParams {
  float A_plus = 0.01f;
  float A_minus = 0.012f;
  float tau_plus = 20.0f;
  float tau_minus = 20.0f;
  float w_max = 1.0f;
};

// Pair-based STDP updater (stateless).
class STDP {
 public:
  // Post spike: update all incoming synapses of post_id.
  static void OnPostSpike(int post_id, float t_post,
                          synaptic::SynapseIndex& synapses,
                          const neural::NeuronPool& pool,
                          const synaptic::SynapseParams& syn_params,
                          const STDPParams& params = {});

  // Pre spike: update all outgoing synapses of pre_id.
  static void OnPreSpike(int pre_id, float t_pre,
                         synaptic::SynapseIndex& synapses,
                         const neural::NeuronPool& pool,
                         const synaptic::SynapseParams& syn_params,
                         const STDPParams& params = {});

  // Supervision: force a post spike at t_post and apply causal updates.
  static void Supervise(int post_id, float t_post,
                        synaptic::SynapseIndex& synapses,
                        neural::NeuronPool& pool,
                        const synaptic::SynapseParams& syn_params,
                        const STDPParams& params = {});

 private:
  static void AdjustWeight(synaptic::Synapse& syn, float delta,
                           const synaptic::SynapseParams& syn_params,
                           const STDPParams& params);
};

}  // namespace senna::plasticity

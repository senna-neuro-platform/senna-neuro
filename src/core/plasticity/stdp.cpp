#include "core/plasticity/stdp.hpp"

#include <algorithm>
#include <cmath>

namespace senna::plasticity {

void STDP::AdjustWeight(synaptic::Synapse& syn, float delta,
                        const synaptic::SynapseParams& syn_params,
                        const STDPParams& params) {
  float w = syn.weight;
  if (delta > 0.0F) {
    // Soft cap near w_max: scale delta as weight approaches bound.
    float taper = std::max(0.0F, 1.0F - w / params.w_max);
    delta *= taper;
  }
  float w_new = w + delta;
  // Clamp magnitude between 0 (or w_min if positive) and w_max.
  float w_min = std::max(0.0F, syn_params.w_min);
  w_new = std::clamp(w_new, w_min, params.w_max);
  syn.weight = w_new;
}

void STDP::OnPostSpike(int post_id, float t_post,
                       synaptic::SynapseIndex& synapses,
                       const neural::NeuronPool& pool,
                       const synaptic::SynapseParams& syn_params,
                       const STDPParams& params) {
  for (auto sid : synapses.Incoming(post_id)) {
    auto& syn = synapses.Get(sid);
    float t_pre = pool.t_spike(syn.pre_id);
    if (t_pre < 0.0F) {
      continue;  // presynaptic neuron never fired
    }
    float delta_t = t_post - t_pre;
    if (delta_t <= 0.0F) {
      continue;  // only causal here; anti handled in pre-spike
    }
    float delta_w = params.A_plus * std::exp(-delta_t / params.tau_plus);
    AdjustWeight(syn, delta_w, syn_params, params);
  }
}

void STDP::OnPreSpike(int pre_id, float t_pre, synaptic::SynapseIndex& synapses,
                      const neural::NeuronPool& pool,
                      const synaptic::SynapseParams& syn_params,
                      const STDPParams& params) {
  for (auto sid : synapses.Outgoing(pre_id)) {
    auto& syn = synapses.Get(sid);
    float t_post = pool.t_spike(syn.post_id);
    if (t_post < 0.0F) {
      continue;  // postsynaptic neuron never fired
    }
    float delta_t = t_post - t_pre;
    if (delta_t >= 0.0F) {
      continue;  // anti-causal only
    }
    float delta_w =
        -params.A_minus * std::exp(-std::abs(delta_t) / params.tau_minus);
    AdjustWeight(syn, delta_w, syn_params, params);
  }
}

void STDP::Supervise(int post_id, float t_post,
                     synaptic::SynapseIndex& synapses, neural::NeuronPool& pool,
                     const synaptic::SynapseParams& syn_params,
                     const STDPParams& params) {
  pool.Fire(post_id, t_post);
  OnPostSpike(post_id, t_post, synapses, pool, syn_params, params);
}

}  // namespace senna::plasticity

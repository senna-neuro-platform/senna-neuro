#include "core/plasticity/stdp.h"

namespace senna::core::plasticity {

STDPRule::STDPRule(const STDPConfig config) : config_(config) { validate_config(); }

void STDPRule::on_pre_spike(const senna::core::domain::NeuronId pre,
                            const senna::core::domain::Time t_pre,
                            senna::core::domain::SynapseStore& synapses) {
    last_pre_spike_[pre] = t_pre;

    for (const auto synapse_id : synapses.outgoing(pre)) {
        auto& synapse = synapses.at(synapse_id);
        const auto last_post = last_post_spike_.find(synapse.post_id);
        if (last_post == last_post_spike_.end()) {
            continue;
        }

        const auto delta_t = last_post->second - t_pre;
        apply_weight_update(synapse, compute_delta_w(delta_t, synapse.weight));
    }
}

void STDPRule::on_post_spike(const senna::core::domain::NeuronId post,
                             const senna::core::domain::Time t_post,
                             senna::core::domain::SynapseStore& synapses) {
    last_post_spike_[post] = t_post;

    for (const auto synapse_id : synapses.incoming(post)) {
        auto& synapse = synapses.at(synapse_id);
        const auto last_pre = last_pre_spike_.find(synapse.pre_id);
        if (last_pre == last_pre_spike_.end()) {
            continue;
        }

        const auto delta_t = t_post - last_pre->second;
        apply_weight_update(synapse, compute_delta_w(delta_t, synapse.weight));
    }
}

void STDPRule::reset_traces() {
    last_pre_spike_.clear();
    last_post_spike_.clear();
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
senna::core::domain::Weight STDPRule::compute_delta_w(
    const senna::core::domain::Time delta_t,
    const senna::core::domain::Weight current_weight) const noexcept {
    if (delta_t > 0.0F) {
        auto delta = config_.a_plus * std::exp(-(delta_t / config_.tau_plus));
        const auto soft_limit = std::max(0.0F, (config_.w_max - current_weight) / config_.w_max);
        delta *= soft_limit;
        return delta;
    }

    if (delta_t < 0.0F) {
        return -config_.a_minus * std::exp(-(std::fabs(delta_t) / config_.tau_minus));
    }

    return 0.0F;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void STDPRule::apply_weight_update(senna::core::domain::Synapse& synapse,
                                   const senna::core::domain::Weight delta_w) const noexcept {
    synapse.weight = std::clamp(synapse.weight + delta_w, 0.0F, config_.w_max);
}

void STDPRule::validate_config() const {
    if (config_.tau_plus <= 0.0F || config_.tau_minus <= 0.0F) {
        throw std::invalid_argument("STDP tau constants must be positive");
    }
    if (config_.w_max <= 0.0F) {
        throw std::invalid_argument("STDP w_max must be positive");
    }
    if (config_.a_plus < 0.0F || config_.a_minus < 0.0F) {
        throw std::invalid_argument("STDP learning rates must be non-negative");
    }
}

}  // namespace senna::core::plasticity

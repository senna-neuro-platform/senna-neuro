#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "core/plasticity/iplasticity_rule.h"

namespace senna::core::plasticity {

struct STDPConfig {
    senna::core::domain::Weight a_plus{0.01F};
    senna::core::domain::Weight a_minus{0.012F};
    senna::core::domain::Time tau_plus{20.0F};
    senna::core::domain::Time tau_minus{20.0F};
    senna::core::domain::Weight w_max{1.0F};
};

class STDPRule final : public IPlasticityRule {
   public:
    explicit STDPRule(STDPConfig config = {});

    void on_pre_spike(senna::core::domain::NeuronId pre, senna::core::domain::Time t_pre,
                      senna::core::domain::SynapseStore& synapses) override;

    void on_post_spike(senna::core::domain::NeuronId post, senna::core::domain::Time t_post,
                       senna::core::domain::SynapseStore& synapses) override;

    void reset_traces();

    [[nodiscard]] const STDPConfig& config() const noexcept { return config_; }

   private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] senna::core::domain::Weight compute_delta_w(
        senna::core::domain::Time delta_t,
        senna::core::domain::Weight current_weight) const noexcept;

    void apply_weight_update(senna::core::domain::Synapse& synapse,
                             senna::core::domain::Weight delta_w) const noexcept;

    void validate_config() const;

    STDPConfig config_{};
    std::unordered_map<senna::core::domain::NeuronId, senna::core::domain::Time> last_pre_spike_{};
    std::unordered_map<senna::core::domain::NeuronId, senna::core::domain::Time> last_post_spike_{};
};

}  // namespace senna::core::plasticity

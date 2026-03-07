#pragma once

#include "core/domain/synapse.h"
#include "core/domain/types.h"

namespace senna::core::plasticity {

class IPlasticityRule {
   public:
    virtual ~IPlasticityRule() = default;

    virtual void on_pre_spike(senna::core::domain::NeuronId pre, senna::core::domain::Time t_pre,
                              senna::core::domain::SynapseStore& synapses) = 0;

    virtual void on_post_spike(senna::core::domain::NeuronId post, senna::core::domain::Time t_post,
                               senna::core::domain::SynapseStore& synapses) = 0;
};

}  // namespace senna::core::plasticity

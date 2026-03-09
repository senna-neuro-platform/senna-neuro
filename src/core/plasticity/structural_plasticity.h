#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/domain/lattice.h"
#include "core/domain/synapse.h"

namespace senna::core::plasticity {

struct StructuralPlasticityConfig {
    senna::core::domain::Weight w_min{0.001F};
    std::size_t update_interval_ticks{10'000U};
    float r_target_hz{5.0F};
    float quiet_ratio{0.5F};
    float sprout_radius{2.0F};
    senna::core::domain::Weight sprout_weight{0.01F};
    senna::core::domain::Time c_base{1.0F};
    std::size_t max_sprouts_per_neuron{1U};
};

struct StructuralPlasticityStats {
    std::size_t pruned{0U};
    std::size_t sprouted{0U};
};

class StructuralPlasticity final {
   public:
    explicit StructuralPlasticity(StructuralPlasticityConfig config = {});

    StructuralPlasticityStats on_tick(const senna::core::domain::Lattice& lattice,
                                      const std::vector<senna::core::domain::Neuron>& neurons,
                                      senna::core::domain::SynapseStore& synapses);

    StructuralPlasticityStats run_once(const senna::core::domain::Lattice& lattice,
                                       const std::vector<senna::core::domain::Neuron>& neurons,
                                       senna::core::domain::SynapseStore& synapses);

    void reset() noexcept;

    [[nodiscard]] const StructuralPlasticityConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::size_t ticks_since_update() const noexcept { return ticks_since_update_; }

    [[nodiscard]] std::size_t total_pruned() const noexcept { return total_pruned_; }

    [[nodiscard]] std::size_t total_sprouted() const noexcept { return total_sprouted_; }

   private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] static bool has_connection(const senna::core::domain::SynapseStore& synapses,
                                             senna::core::domain::NeuronId pre_id,
                                             senna::core::domain::NeuronId post_id);

    std::size_t prune_weak_synapses(senna::core::domain::SynapseStore& synapses) const;

    std::size_t sprout_for_quiet_neurons(const senna::core::domain::Lattice& lattice,
                                         const std::vector<senna::core::domain::Neuron>& neurons,
                                         senna::core::domain::SynapseStore& synapses) const;

    void validate_config() const;

    StructuralPlasticityConfig config_{};
    std::size_t ticks_since_update_{0U};
    std::size_t total_pruned_{0U};
    std::size_t total_sprouted_{0U};
};

}  // namespace senna::core::plasticity

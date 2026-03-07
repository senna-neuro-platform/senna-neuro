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
    explicit StructuralPlasticity(const StructuralPlasticityConfig config = {}) : config_(config) {
        validate_config();
    }

    StructuralPlasticityStats on_tick(const senna::core::domain::Lattice& lattice,
                                      const std::vector<senna::core::domain::Neuron>& neurons,
                                      senna::core::domain::SynapseStore& synapses) {
        ++ticks_since_update_;
        if (ticks_since_update_ < config_.update_interval_ticks) {
            return {};
        }

        ticks_since_update_ = 0U;
        return run_once(lattice, neurons, synapses);
    }

    StructuralPlasticityStats run_once(const senna::core::domain::Lattice& lattice,
                                       const std::vector<senna::core::domain::Neuron>& neurons,
                                       senna::core::domain::SynapseStore& synapses) {
        if (neurons.size() != lattice.neuron_count()) {
            throw std::invalid_argument("StructuralPlasticity expects neurons.size() == lattice.neuron_count()");
        }

        StructuralPlasticityStats stats{};

        stats.pruned = prune_weak_synapses(synapses);
        synapses.rebuild_indices(lattice.neuron_count());

        stats.sprouted = sprout_for_quiet_neurons(lattice, neurons, synapses);
        synapses.rebuild_indices(lattice.neuron_count());

        total_pruned_ += stats.pruned;
        total_sprouted_ += stats.sprouted;
        return stats;
    }

    void reset() noexcept {
        ticks_since_update_ = 0U;
        total_pruned_ = 0U;
        total_sprouted_ = 0U;
    }

    [[nodiscard]] const StructuralPlasticityConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::size_t ticks_since_update() const noexcept { return ticks_since_update_; }

    [[nodiscard]] std::size_t total_pruned() const noexcept { return total_pruned_; }

    [[nodiscard]] std::size_t total_sprouted() const noexcept { return total_sprouted_; }

   private:
    [[nodiscard]] static bool has_connection(const senna::core::domain::SynapseStore& synapses,
                                             const senna::core::domain::NeuronId pre_id,
                                             const senna::core::domain::NeuronId post_id) {
        for (const auto synapse_id : synapses.outgoing(pre_id)) {
            if (synapses.at(synapse_id).post_id == post_id) {
                return true;
            }
        }
        return false;
    }

    std::size_t prune_weak_synapses(senna::core::domain::SynapseStore& synapses) const {
        auto& all_synapses = synapses.synapses();
        const auto before = all_synapses.size();

        all_synapses.erase(
            std::remove_if(
                all_synapses.begin(), all_synapses.end(),
                [w_min = config_.w_min](const senna::core::domain::Synapse& synapse) {
                    return std::fabs(synapse.weight) < w_min;
                }),
            all_synapses.end());

        return before - all_synapses.size();
    }

    std::size_t sprout_for_quiet_neurons(const senna::core::domain::Lattice& lattice,
                                         const std::vector<senna::core::domain::Neuron>& neurons,
                                         senna::core::domain::SynapseStore& synapses) const {
        std::size_t sprouted = 0U;
        const auto quiet_threshold = config_.r_target_hz * config_.quiet_ratio;

        for (const auto& post_neuron : neurons) {
            if (post_neuron.average_rate() >= quiet_threshold) {
                continue;
            }

            std::size_t sprouts_for_neuron = 0U;
            const auto candidates = lattice.neighbors(post_neuron.id(), config_.sprout_radius);

            for (const auto& candidate : candidates) {
                if (sprouts_for_neuron >= config_.max_sprouts_per_neuron) {
                    break;
                }

                const auto pre_id = candidate.id;
                const auto post_id = post_neuron.id();
                if (has_connection(synapses, pre_id, post_id)) {
                    continue;
                }

                const auto& pre_neuron = neurons.at(static_cast<std::size_t>(pre_id));
                synapses.connect(pre_id, post_id, pre_neuron.position(), post_neuron.position(),
                                pre_neuron.type(), config_.sprout_weight, config_.c_base);
                ++sprouts_for_neuron;
                ++sprouted;
            }
        }

        return sprouted;
    }

    void validate_config() const {
        if (config_.w_min < 0.0F) {
            throw std::invalid_argument("StructuralPlasticity w_min must be non-negative");
        }
        if (config_.update_interval_ticks == 0U) {
            throw std::invalid_argument("StructuralPlasticity update_interval_ticks must be >= 1");
        }
        if (config_.r_target_hz < 0.0F) {
            throw std::invalid_argument("StructuralPlasticity r_target_hz must be non-negative");
        }
        if (config_.quiet_ratio <= 0.0F || config_.quiet_ratio > 1.0F) {
            throw std::invalid_argument("StructuralPlasticity quiet_ratio must be in (0, 1]");
        }
        if (config_.sprout_radius <= 0.0F) {
            throw std::invalid_argument("StructuralPlasticity sprout_radius must be positive");
        }
        if (config_.sprout_weight <= 0.0F) {
            throw std::invalid_argument("StructuralPlasticity sprout_weight must be positive");
        }
        if (config_.c_base <= 0.0F) {
            throw std::invalid_argument("StructuralPlasticity c_base must be positive");
        }
        if (config_.max_sprouts_per_neuron == 0U) {
            throw std::invalid_argument("StructuralPlasticity max_sprouts_per_neuron must be >= 1");
        }
    }

    StructuralPlasticityConfig config_{};
    std::size_t ticks_since_update_{0U};
    std::size_t total_pruned_{0U};
    std::size_t total_sprouted_{0U};
};

}  // namespace senna::core::plasticity

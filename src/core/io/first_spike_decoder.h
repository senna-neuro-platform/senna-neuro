#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/io/idecoder.h"

namespace senna::core::io {

class FirstSpikeDecoder final : public IDecoder {
   public:
    explicit FirstSpikeDecoder(std::vector<senna::core::domain::NeuronId> output_neurons,
                               const senna::core::domain::Weight wta_weight = 10.0F)
        : output_neurons_(std::move(output_neurons)), wta_weight_(-std::abs(wta_weight)) {
        if (output_neurons_.empty()) {
            throw std::invalid_argument("FirstSpikeDecoder requires non-empty output neurons");
        }

        output_index_.reserve(output_neurons_.size());
        for (std::size_t index = 0U; index < output_neurons_.size(); ++index) {
            const auto [_, inserted] = output_index_.emplace(output_neurons_[index], index);
            if (!inserted) {
                throw std::invalid_argument("FirstSpikeDecoder output neuron list must be unique");
            }
        }
    }

    [[nodiscard]] int decode(
        const std::vector<senna::core::domain::SpikeEvent>& output_spikes) const override {
        constexpr float kEpsilon = 1e-6F;
        auto best_time = std::numeric_limits<senna::core::domain::Time>::infinity();
        std::size_t best_index = output_neurons_.size();

        for (const auto& spike : output_spikes) {
            const auto it = output_index_.find(spike.source);
            if (it == output_index_.end()) {
                continue;
            }

            const auto index = it->second;
            if ((spike.arrival + kEpsilon) < best_time) {
                best_time = spike.arrival;
                best_index = index;
                continue;
            }

            if (std::fabs(spike.arrival - best_time) <= kEpsilon && index < best_index) {
                best_index = index;
            }
        }

        if (best_index == output_neurons_.size()) {
            return -1;
        }
        return static_cast<int>(best_index);
    }

    [[nodiscard]] std::vector<senna::core::domain::SpikeEvent> winner_take_all_events(
        const senna::core::domain::NeuronId winner_id,
        const senna::core::domain::Time t_now) const {
        std::vector<senna::core::domain::SpikeEvent> inhibitory{};
        if (!output_index_.contains(winner_id)) {
            return inhibitory;
        }

        inhibitory.reserve(output_neurons_.size() - 1U);
        for (const auto output_id : output_neurons_) {
            if (output_id == winner_id) {
                continue;
            }
            inhibitory.push_back(senna::core::domain::SpikeEvent{
                winner_id,
                output_id,
                t_now,
                wta_weight_,
            });
        }
        return inhibitory;
    }

    [[nodiscard]] const std::vector<senna::core::domain::NeuronId>& output_neurons()
        const noexcept {
        return output_neurons_;
    }

    [[nodiscard]] senna::core::domain::Weight wta_weight() const noexcept { return wta_weight_; }

   private:
    std::vector<senna::core::domain::NeuronId> output_neurons_{};
    senna::core::domain::Weight wta_weight_{-10.0F};
    std::unordered_map<senna::core::domain::NeuronId, std::size_t> output_index_{};
};

}  // namespace senna::core::io

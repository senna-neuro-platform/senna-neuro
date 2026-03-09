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
                               senna::core::domain::Weight wta_weight = 10.0F);

    [[nodiscard]] int decode(
        const std::vector<senna::core::domain::SpikeEvent>& output_spikes) const override;

    [[nodiscard]] std::vector<senna::core::domain::SpikeEvent> winner_take_all_events(
        senna::core::domain::NeuronId winner_id, senna::core::domain::Time t_now) const;

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

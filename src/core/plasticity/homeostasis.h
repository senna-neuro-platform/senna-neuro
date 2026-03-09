#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/types.h"

namespace senna::core::plasticity {

struct HomeostasisConfig {
    float alpha{0.999F};
    float r_target_hz{5.0F};
    float eta_homeo{0.001F};
    senna::core::domain::Voltage theta_min{0.1F};
    senna::core::domain::Voltage theta_max{5.0F};
    std::size_t update_interval_ticks{1U};
};

class Homeostasis final {
   public:
    explicit Homeostasis(HomeostasisConfig config = {});

    void on_spike(const senna::core::domain::SpikeEvent& spike);

    void on_tick(std::vector<senna::core::domain::Neuron>& neurons,
                 senna::core::domain::Time dt_ms);

    void reset();

    [[nodiscard]] const HomeostasisConfig& config() const noexcept { return config_; }

   private:
    void ensure_capacity(std::size_t size);

    void validate_config() const;

    HomeostasisConfig config_{};
    std::vector<std::uint32_t> spike_counts_{};
    std::size_t ticks_since_update_{0U};
};

}  // namespace senna::core::plasticity

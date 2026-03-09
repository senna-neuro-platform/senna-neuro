#include "core/plasticity/homeostasis.h"

namespace senna::core::plasticity {

Homeostasis::Homeostasis(const HomeostasisConfig config) : config_(config) { validate_config(); }

void Homeostasis::on_spike(const senna::core::domain::SpikeEvent& spike) {
    const auto index = static_cast<std::size_t>(spike.source);
    ensure_capacity(index + 1U);
    ++spike_counts_[index];
}

void Homeostasis::on_tick(std::vector<senna::core::domain::Neuron>& neurons,
                          const senna::core::domain::Time dt_ms) {
    if (dt_ms <= 0.0F) {
        throw std::invalid_argument("Homeostasis dt_ms must be positive");
    }

    ensure_capacity(neurons.size());
    ++ticks_since_update_;
    if (ticks_since_update_ < config_.update_interval_ticks) {
        return;
    }

    const auto window_ticks = static_cast<float>(ticks_since_update_);
    const auto alpha_window = std::pow(config_.alpha, window_ticks);
    const auto window_ms = dt_ms * window_ticks;
    const auto window_seconds = window_ms / 1000.0F;

    for (std::size_t neuron_index = 0U; neuron_index < neurons.size(); ++neuron_index) {
        auto& neuron = neurons[neuron_index];

        const auto spike_count = static_cast<float>(spike_counts_[neuron_index]);
        const auto observed_rate_hz = spike_count / window_seconds;

        const auto updated_rate =
            (alpha_window * neuron.average_rate()) + ((1.0F - alpha_window) * observed_rate_hz);
        neuron.set_average_rate(updated_rate);

        const auto theta_delta = config_.eta_homeo * (neuron.average_rate() - config_.r_target_hz);
        neuron.adjust_threshold(theta_delta, config_.theta_min, config_.theta_max);

        spike_counts_[neuron_index] = 0U;
    }

    ticks_since_update_ = 0U;
}

void Homeostasis::reset() {
    std::fill(spike_counts_.begin(), spike_counts_.end(), 0U);
    ticks_since_update_ = 0U;
}

void Homeostasis::ensure_capacity(const std::size_t size) {
    if (spike_counts_.size() < size) {
        spike_counts_.resize(size, 0U);
    }
}

void Homeostasis::validate_config() const {
    if (config_.alpha < 0.0F || config_.alpha >= 1.0F) {
        throw std::invalid_argument("Homeostasis alpha must be in [0, 1)");
    }
    if (config_.r_target_hz < 0.0F) {
        throw std::invalid_argument("Homeostasis r_target_hz must be non-negative");
    }
    if (config_.eta_homeo < 0.0F) {
        throw std::invalid_argument("Homeostasis eta_homeo must be non-negative");
    }
    if (config_.theta_min < 0.0F || config_.theta_max < 0.0F) {
        throw std::invalid_argument("Homeostasis theta bounds must be non-negative");
    }
    if (config_.theta_min > config_.theta_max) {
        throw std::invalid_argument("Homeostasis theta_min must be <= theta_max");
    }
    if (config_.update_interval_ticks == 0U) {
        throw std::invalid_argument("Homeostasis update_interval_ticks must be >= 1");
    }
}

}  // namespace senna::core::plasticity

#include "core/domain/neuron.h"

namespace senna::core::domain {

Neuron::Neuron(NeuronId id, Coord3D position, NeuronType type, NeuronConfig config) noexcept
    : id_(id),
      position_(position),
      type_(type),
      V_(config.v_rest),
      theta_(config.theta_base),
      t_last_(0.0F),
      t_spike_(-std::numeric_limits<Time>::infinity()),
      r_avg_(0.0F),
      in_refractory_(false),
      config_(config) {}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::optional<SpikeEvent> Neuron::receive_input(const Time t_now, const Weight input) noexcept {
    Time effective_now = t_now;
    if (effective_now < t_last_) {
        effective_now = t_last_;
    }

    in_refractory_ = effective_now < (t_spike_ + config_.t_ref);
    if (in_refractory_) {
        return std::nullopt;
    }

    const Time dt = effective_now - t_last_;
    if (dt > 0.0F) {
        const Time safe_tau_m = config_.tau_m > 0.0F ? config_.tau_m : 1.0F;
        const auto decay = std::exp(-(dt / safe_tau_m));
        V_ = config_.v_rest + ((V_ - config_.v_rest) * decay);
    }

    V_ += input;
    t_last_ = effective_now;

    if (V_ < theta_) {
        return std::nullopt;
    }

    V_ = config_.v_reset;
    t_spike_ = effective_now;
    in_refractory_ = true;

    return SpikeEvent{id_, id_, effective_now, spike_value()};
}

Neuron Neuron::from_snapshot(const NeuronSnapshot& state) noexcept {
    Neuron neuron{state.id, state.position, state.type, state.config};
    neuron.restore_from_snapshot(state);
    return neuron;
}

void Neuron::restore_from_snapshot(const NeuronSnapshot& state) noexcept {
    id_ = state.id;
    position_ = state.position;
    type_ = state.type;
    config_ = state.config;
    V_ = state.potential;
    theta_ = state.threshold;
    t_last_ = state.last_update_time;
    t_spike_ = state.last_spike_time;
    r_avg_ = state.average_rate;
    in_refractory_ = state.in_refractory;
}

void Neuron::set_threshold(const Voltage threshold) noexcept { theta_ = threshold; }

void Neuron::adjust_threshold(const Voltage delta, const Voltage theta_min,
                              const Voltage theta_max) noexcept {
    theta_ = std::clamp(theta_ + delta, theta_min, theta_max);
}

void Neuron::set_average_rate(const float rate_hz) noexcept { r_avg_ = std::max(0.0F, rate_hz); }

}  // namespace senna::core::domain

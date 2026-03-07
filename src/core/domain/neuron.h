#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "core/domain/types.h"

namespace senna::core::domain {

struct NeuronConfig {
    Voltage v_rest{0.0F};
    Voltage v_reset{0.0F};
    Time tau_m{20.0F};
    Time t_ref{2.0F};
    Voltage theta_base{1.0F};
};

struct NeuronSnapshot {
    NeuronId id{};
    Coord3D position{};
    NeuronType type{NeuronType::Excitatory};
    NeuronConfig config{};
    Voltage potential{0.0F};
    Voltage threshold{1.0F};
    Time last_update_time{0.0F};
    Time last_spike_time{-std::numeric_limits<Time>::infinity()};
    float average_rate{0.0F};
    bool in_refractory{false};
};

class Neuron final {
   public:
    explicit Neuron(NeuronId id, Coord3D position, NeuronType type,
                    NeuronConfig config = {}) noexcept
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

    [[nodiscard]] std::optional<SpikeEvent> receive_input(Time t_now, Weight input) noexcept {
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

    [[nodiscard]] NeuronId id() const noexcept { return id_; }
    [[nodiscard]] const Coord3D& position() const noexcept { return position_; }
    [[nodiscard]] NeuronType type() const noexcept { return type_; }
    [[nodiscard]] Voltage potential() const noexcept { return V_; }
    [[nodiscard]] Voltage threshold() const noexcept { return theta_; }
    [[nodiscard]] Time last_update_time() const noexcept { return t_last_; }
    [[nodiscard]] Time last_spike_time() const noexcept { return t_spike_; }
    [[nodiscard]] float average_rate() const noexcept { return r_avg_; }
    [[nodiscard]] bool in_refractory() const noexcept { return in_refractory_; }
    [[nodiscard]] const NeuronConfig& config() const noexcept { return config_; }
    [[nodiscard]] NeuronSnapshot snapshot() const noexcept {
        return NeuronSnapshot{
            id_, position_, type_, config_, V_, theta_, t_last_, t_spike_, r_avg_, in_refractory_,
        };
    }

    [[nodiscard]] static Neuron from_snapshot(const NeuronSnapshot& state) noexcept {
        Neuron neuron{state.id, state.position, state.type, state.config};
        neuron.restore_from_snapshot(state);
        return neuron;
    }

    void restore_from_snapshot(const NeuronSnapshot& state) noexcept {
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

    void set_threshold(const Voltage threshold) noexcept { theta_ = threshold; }

    void adjust_threshold(const Voltage delta, const Voltage theta_min,
                          const Voltage theta_max) noexcept {
        theta_ = std::clamp(theta_ + delta, theta_min, theta_max);
    }

    void set_average_rate(const float rate_hz) noexcept { r_avg_ = std::max(0.0F, rate_hz); }

   private:
    [[nodiscard]] Weight spike_value() const noexcept {
        return type_ == NeuronType::Excitatory ? 1.0F : -1.0F;
    }

    NeuronId id_{};
    Coord3D position_{};
    NeuronType type_{NeuronType::Excitatory};
    Voltage V_{0.0F};
    Voltage theta_{1.0F};
    Time t_last_{0.0F};
    Time t_spike_{-std::numeric_limits<Time>::infinity()};
    float r_avg_{0.0F};
    bool in_refractory_{false};
    NeuronConfig config_{};
};

}  // namespace senna::core::domain

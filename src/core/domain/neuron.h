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
                    NeuronConfig config = {}) noexcept;

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    [[nodiscard]] std::optional<SpikeEvent> receive_input(Time t_now, Weight input) noexcept;

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

    [[nodiscard]] static Neuron from_snapshot(const NeuronSnapshot& state) noexcept;

    void restore_from_snapshot(const NeuronSnapshot& state) noexcept;

    void set_threshold(Voltage threshold) noexcept;

    void adjust_threshold(Voltage delta, Voltage theta_min, Voltage theta_max) noexcept;

    void set_average_rate(float rate_hz) noexcept;

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

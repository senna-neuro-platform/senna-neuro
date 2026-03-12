#include "core/config/runtime_config.hpp"

#include <yaml-cpp/yaml.h>

namespace senna::config {

namespace {
template <typename T>
T MaybeGet(const YAML::Node& node, const char* key, const T& def) {
  if (!node || !node[key]) return def;
  try {
    return node[key].as<T>();
  } catch (...) {
    return def;
  }
}
}  // namespace

RuntimeConfig LoadRuntimeConfig(const std::string& path) {
  RuntimeConfig cfg;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (...) {
    return cfg;  // fallback to defaults
  }

  if (auto sim = root["simulation"]) {
    cfg.network.dt = MaybeGet<float>(sim, "dt", cfg.network.dt);
    cfg.network.seed = MaybeGet<uint64_t>(sim, "seed", cfg.network.seed);
  }

  if (auto lattice = root["lattice"]) {
    cfg.network.width = MaybeGet<int>(lattice, "width", cfg.network.width);
    cfg.network.height = MaybeGet<int>(lattice, "height", cfg.network.height);
    cfg.network.depth = MaybeGet<int>(lattice, "depth", cfg.network.depth);
    cfg.network.density =
        MaybeGet<double>(lattice, "density", cfg.network.density);
    cfg.network.neighbor_radius = MaybeGet<float>(lattice, "neighbor_radius",
                                                  cfg.network.neighbor_radius);
    cfg.network.num_outputs =
        MaybeGet<int>(lattice, "num_outputs", cfg.network.num_outputs);
    cfg.network.excitatory_ratio = MaybeGet<double>(
        lattice, "excitatory_ratio", cfg.network.excitatory_ratio);
  }

  if (auto lif = root["lif"]) {
    auto& p = cfg.network.lif_params;
    p.V_rest = MaybeGet<float>(lif, "V_rest", p.V_rest);
    p.V_reset = MaybeGet<float>(lif, "V_reset", p.V_reset);
    p.tau_m = MaybeGet<float>(lif, "tau_m", p.tau_m);
    p.t_ref = MaybeGet<float>(lif, "t_ref", p.t_ref);
    p.theta_base = MaybeGet<float>(lif, "theta_base", p.theta_base);
  }

  if (auto syn = root["synapse"]) {
    auto& p = cfg.network.synapse_params;
    p.w_min = MaybeGet<float>(syn, "w_min", p.w_min);
    p.w_max = MaybeGet<float>(syn, "w_max", p.w_max);
    p.c_base = MaybeGet<float>(syn, "c_base", p.c_base);
    p.w_wta = MaybeGet<float>(syn, "w_wta", p.w_wta);
  }

  if (auto stdp = root["stdp"]) {
    cfg.network.stdp_params.A_plus =
        MaybeGet<float>(stdp, "A_plus", cfg.network.stdp_params.A_plus);
    cfg.network.stdp_params.A_minus =
        MaybeGet<float>(stdp, "A_minus", cfg.network.stdp_params.A_minus);
    cfg.network.stdp_params.tau_plus =
        MaybeGet<float>(stdp, "tau_plus", cfg.network.stdp_params.tau_plus);
    cfg.network.stdp_params.tau_minus =
        MaybeGet<float>(stdp, "tau_minus", cfg.network.stdp_params.tau_minus);
    cfg.network.stdp_params.w_max =
        MaybeGet<float>(stdp, "w_max", cfg.network.stdp_params.w_max);
  }

  if (auto homeo = root["homeostasis"]) {
    cfg.network.homeostasis.alpha =
        MaybeGet<float>(homeo, "alpha", cfg.network.homeostasis.alpha);
    cfg.network.homeostasis.target_rate_hz = MaybeGet<float>(
        homeo, "target_rate", cfg.network.homeostasis.target_rate_hz);
    cfg.network.homeostasis.theta_step = MaybeGet<float>(
        homeo, "theta_step", cfg.network.homeostasis.theta_step);
    cfg.network.homeostasis.theta_min =
        MaybeGet<float>(homeo, "theta_min", cfg.network.homeostasis.theta_min);
    cfg.network.homeostasis.theta_max =
        MaybeGet<float>(homeo, "theta_max", cfg.network.homeostasis.theta_max);
    cfg.network.homeostasis.interval_ticks = MaybeGet<int>(
        homeo, "interval_ticks", cfg.network.homeostasis.interval_ticks);
  }

  if (auto enc = root["encoder"]) {
    cfg.network.encoder_params.max_rate =
        MaybeGet<float>(enc, "max_rate", cfg.network.encoder_params.max_rate);
    cfg.network.encoder_params.presentation_ms = MaybeGet<float>(
        enc, "presentation_ms", cfg.network.encoder_params.presentation_ms);
    cfg.network.encoder_params.input_value = MaybeGet<float>(
        enc, "input_value", cfg.network.encoder_params.input_value);
  }

  if (auto dec = root["decoder"]) {
    cfg.network.decoder_window_ms =
        MaybeGet<float>(dec, "window_ms", cfg.network.decoder_window_ms);
  }

  return cfg;
}

}  // namespace senna::config

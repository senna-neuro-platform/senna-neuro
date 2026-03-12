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

  if (auto homeo = root["homeostasis"]) {
    cfg.homeostasis.alpha =
        MaybeGet<float>(homeo, "alpha", cfg.homeostasis.alpha);
    cfg.homeostasis.target_rate =
        MaybeGet<float>(homeo, "target_rate", cfg.homeostasis.target_rate);
    cfg.homeostasis.theta_step =
        MaybeGet<float>(homeo, "theta_step", cfg.homeostasis.theta_step);
  }

  if (auto enc = root["encoder"]) {
    cfg.encoder.max_rate =
        MaybeGet<float>(enc, "max_rate", cfg.encoder.max_rate);
    cfg.encoder.presentation_ms =
        MaybeGet<float>(enc, "presentation_ms", cfg.encoder.presentation_ms);
    cfg.encoder.input_value =
        MaybeGet<float>(enc, "input_value", cfg.encoder.input_value);
  }

  if (auto dec = root["decoder"]) {
    cfg.decoder_window_ms =
        MaybeGet<float>(dec, "window_ms", cfg.decoder_window_ms);
  }

  return cfg;
}

}  // namespace senna::config

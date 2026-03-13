#include "core/config/runtime_config.hpp"

#include <yaml-cpp/yaml.h>

namespace senna::config {

namespace {
template <typename T>
T GetOrDefault(const YAML::Node& node, const char* key, const T& def) {
  if (!node || !node[key]) {
    return def;
  }
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
    cfg.network.dt = GetOrDefault<float>(sim, "dt", cfg.network.dt);
    cfg.network.seed = GetOrDefault<uint64_t>(sim, "seed", cfg.network.seed);
  }

  if (auto lattice = root["lattice"]) {
    cfg.network.width = GetOrDefault<int>(lattice, "width", cfg.network.width);
    cfg.network.height =
        GetOrDefault<int>(lattice, "height", cfg.network.height);
    cfg.network.depth = GetOrDefault<int>(lattice, "depth", cfg.network.depth);
    cfg.network.density =
        GetOrDefault<double>(lattice, "density", cfg.network.density);
    cfg.network.neighbor_radius = GetOrDefault<float>(
        lattice, "neighbor_radius", cfg.network.neighbor_radius);
    cfg.network.num_outputs =
        GetOrDefault<int>(lattice, "num_outputs", cfg.network.num_outputs);
    cfg.network.excitatory_ratio = GetOrDefault<double>(
        lattice, "excitatory_ratio", cfg.network.excitatory_ratio);
  }

  if (auto lif = root["lif"]) {
    auto& p = cfg.network.lif_params;
    p.V_rest = GetOrDefault<float>(lif, "V_rest", p.V_rest);
    p.V_reset = GetOrDefault<float>(lif, "V_reset", p.V_reset);
    p.tau_m = GetOrDefault<float>(lif, "tau_m", p.tau_m);
    p.t_ref = GetOrDefault<float>(lif, "t_ref", p.t_ref);
    p.theta_base = GetOrDefault<float>(lif, "theta_base", p.theta_base);
  }

  if (auto syn = root["synapse"]) {
    auto& p = cfg.network.synapse_params;
    p.w_min = GetOrDefault<float>(syn, "w_min", p.w_min);
    p.w_max = GetOrDefault<float>(syn, "w_max", p.w_max);
    p.c_base = GetOrDefault<float>(syn, "c_base", p.c_base);
    p.w_wta = GetOrDefault<float>(syn, "w_wta", p.w_wta);
  }

  if (auto stdp = root["stdp"]) {
    cfg.network.stdp_params.A_plus =
        GetOrDefault<float>(stdp, "A_plus", cfg.network.stdp_params.A_plus);
    cfg.network.stdp_params.A_minus =
        GetOrDefault<float>(stdp, "A_minus", cfg.network.stdp_params.A_minus);
    cfg.network.stdp_params.tau_plus =
        GetOrDefault<float>(stdp, "tau_plus", cfg.network.stdp_params.tau_plus);
    cfg.network.stdp_params.tau_minus = GetOrDefault<float>(
        stdp, "tau_minus", cfg.network.stdp_params.tau_minus);
    cfg.network.stdp_params.w_max =
        GetOrDefault<float>(stdp, "w_max", cfg.network.stdp_params.w_max);
  }

  if (auto homeo = root["homeostasis"]) {
    cfg.network.homeostasis.alpha =
        GetOrDefault<float>(homeo, "alpha", cfg.network.homeostasis.alpha);
    cfg.network.homeostasis.target_rate_hz = GetOrDefault<float>(
        homeo, "target_rate", cfg.network.homeostasis.target_rate_hz);
    cfg.network.homeostasis.theta_step = GetOrDefault<float>(
        homeo, "theta_step", cfg.network.homeostasis.theta_step);
    cfg.network.homeostasis.theta_min = GetOrDefault<float>(
        homeo, "theta_min", cfg.network.homeostasis.theta_min);
    cfg.network.homeostasis.theta_max = GetOrDefault<float>(
        homeo, "theta_max", cfg.network.homeostasis.theta_max);
    cfg.network.homeostasis.interval_ticks = GetOrDefault<int>(
        homeo, "interval_ticks", cfg.network.homeostasis.interval_ticks);
    cfg.network.homeostasis.global_mix = GetOrDefault<float>(
        homeo, "global_mix", cfg.network.homeostasis.global_mix);
  }

  if (auto enc = root["encoder"]) {
    cfg.network.encoder_params.max_rate = GetOrDefault<float>(
        enc, "max_rate", cfg.network.encoder_params.max_rate);
    cfg.network.encoder_params.presentation_ms = GetOrDefault<float>(
        enc, "presentation_ms", cfg.network.encoder_params.presentation_ms);
    cfg.network.encoder_params.input_value = GetOrDefault<float>(
        enc, "input_value", cfg.network.encoder_params.input_value);
  }

  if (auto dec = root["decoder"]) {
    cfg.network.decoder_window_ms =
        GetOrDefault<float>(dec, "window_ms", cfg.network.decoder_window_ms);
    cfg.network.decoder_seed =
        GetOrDefault<uint64_t>(dec, "seed", cfg.network.decoder_seed);
    cfg.decoder_cfg.seed = cfg.network.decoder_seed;
  }

  if (auto str = root["structural"]) {
    cfg.network.structural.w_min_prune = GetOrDefault<float>(
        str, "w_min_prune", cfg.network.structural.w_min_prune);
    cfg.network.structural.interval_ticks = GetOrDefault<int>(
        str, "interval_ticks", cfg.network.structural.interval_ticks);
    cfg.network.structural.sprout_radius = GetOrDefault<float>(
        str, "sprout_radius", cfg.network.structural.sprout_radius);
    cfg.network.structural.sprout_weight = GetOrDefault<float>(
        str, "sprout_weight", cfg.network.structural.sprout_weight);
    cfg.network.structural.quiet_fraction = GetOrDefault<float>(
        str, "quiet_fraction", cfg.network.structural.quiet_fraction);
  }

  if (auto ports = root["ports"]) {
    cfg.ports.grpc = GetOrDefault<int>(ports, "grpc", cfg.ports.grpc);
    cfg.ports.ws = GetOrDefault<int>(ports, "ws", cfg.ports.ws);
    cfg.ports.metrics = GetOrDefault<int>(ports, "metrics", cfg.ports.metrics);
  }

  if (auto loop = root["loop"]) {
    cfg.loop_sleep_ms = GetOrDefault<int>(loop, "sleep_ms", cfg.loop_sleep_ms);
  }

  if (auto obs = root["observability"]) {
    if (auto buckets = obs["tick_duration_buckets"]) {
      cfg.observability.tick_duration_buckets = buckets.as<std::vector<double>>(
          cfg.observability.tick_duration_buckets);
    }
    cfg.observability.exporter_backlog = GetOrDefault<int>(
        obs, "exporter_backlog", cfg.observability.exporter_backlog);
  }

  if (auto trainer = root["trainer"]) {
    cfg.trainer.host =
        GetOrDefault<std::string>(trainer, "host", cfg.trainer.host);
    cfg.trainer.port = GetOrDefault<int>(trainer, "port", cfg.trainer.port);
  } else {
    // Align trainer port with gRPC port when not explicitly provided.
    cfg.trainer.port = cfg.ports.grpc;
  }

  // If trainer section exists but omits port, fall back to gRPC port for
  // consistency with runtime bindings.
  if (auto trainer = root["trainer"]) {
    if (!trainer["port"]) {
      cfg.trainer.port = cfg.ports.grpc;
    }
  }

  return cfg;
}

}  // namespace senna::config

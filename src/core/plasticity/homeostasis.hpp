#pragma once

#include <cstdint>
#include <vector>

#include "core/neural/neuron_pool.hpp"

namespace senna::plasticity {

struct HomeostasisConfig {
  float alpha = 0.999f;
  float target_rate_hz = 5.0f;  // desired firing rate (Hz)
  float theta_step = 0.001f;    // learning rate (delta theta per Hz error)
  float theta_min = 0.1f;       // lower bound for threshold
  float theta_max = 5.0f;       // upper bound for threshold
  int interval_ticks = 10;      // how often to run homeostasis
  float global_mix = 0.5f;      // weight of global activity in [0,1]
};

// Stateless homeostasis updater operating on a NeuronPool.
class Homeostasis {
 public:
  explicit Homeostasis(HomeostasisConfig cfg = {}) : cfg_(cfg) {}

  void SetConfig(const HomeostasisConfig& cfg) { cfg_ = cfg; }
  const HomeostasisConfig& config() const { return cfg_; }

  // Compute new thresholds based on provided snapshots.
  // theta_cur: current thresholds snapshot.
  // r_avg: smoothed firing rates snapshot.
  // dt_ms: simulation step (ms) used to convert r_avg to Hz.
  // global_activity: optional fraction of neurons firing this tick ([-1,1]);
  //                  if negative, only local activity is used.
  std::vector<float> ComputeTheta(const std::vector<float>& theta_cur,
                                  const std::vector<float>& r_avg, float dt_ms,
                                  float global_activity) const;

 private:
  HomeostasisConfig cfg_;
};

}  // namespace senna::plasticity

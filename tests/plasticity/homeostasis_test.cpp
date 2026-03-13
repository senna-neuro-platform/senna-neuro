#include "core/plasticity/homeostasis.hpp"

#include <gtest/gtest.h>

namespace senna::plasticity {

// Simple deterministic frequency model: firing rate (Hz) decreases
// linearly with theta; used to approximate closed-loop convergence.
static float ModelFreqHz(float theta) {
  float freq = 10.0F - 2.0F * theta;  // 10 Hz at theta=1, 0 Hz at theta=5
  return freq < 0.0F ? 0.0F : freq;
}

TEST(HomeostasisTest, ConvergesTowardTargetRate) {
  HomeostasisConfig cfg;
  cfg.alpha = 0.99F;
  cfg.target_rate_hz = 5.0F;
  cfg.theta_step = 0.05F;
  cfg.theta_min = 0.1F;
  cfg.theta_max = 5.0F;

  Homeostasis homeo(cfg);
  float dt_ms = 1.0F;

  // Start below target (high freq -> theta should grow, freq should drop).
  std::vector<float> theta = {1.0F};
  std::vector<float> r_avg(1, 0.0F);

  for (int t = 0; t < 1000; ++t) {
    // Update synthetic r_avg based on current theta.
    float freq = ModelFreqHz(theta[0]);
    float activity = freq * (dt_ms * 1e-3F);  // convert Hz to per-tick prob
    r_avg[0] = cfg.alpha * r_avg[0] + (1.0F - cfg.alpha) * activity;

    auto theta_new = homeo.ComputeTheta(theta, r_avg, dt_ms,
                                        /*global_activity=*/-1.0F);
    theta.swap(theta_new);
  }

  float freq_final = ModelFreqHz(theta[0]);
  EXPECT_NEAR(freq_final, cfg.target_rate_hz, 0.5F);
  EXPECT_GE(theta[0], cfg.theta_min);
  EXPECT_LE(theta[0], cfg.theta_max);
}

TEST(HomeostasisTest, RespectsThetaBounds) {
  HomeostasisConfig cfg;
  cfg.alpha = 0.9F;
  cfg.target_rate_hz = 100.0F;  // force downward adjustment
  cfg.theta_step = 10.0F;
  cfg.theta_min = 0.5F;
  cfg.theta_max = 1.0F;
  cfg.global_mix = 0.0F;  // use local only to isolate effect

  Homeostasis homeo(cfg);
  float dt_ms = 1.0F;

  std::vector<float> theta = {0.6F};
  std::vector<float> r_avg = {0.0F};  // very low activity vs huge target
  auto theta_new = homeo.ComputeTheta(theta, r_avg, dt_ms,
                                      /*global_activity=*/-1.0F);
  EXPECT_GE(theta_new[0], cfg.theta_min);
  EXPECT_LE(theta_new[0], cfg.theta_max);
}

TEST(HomeostasisTest, GlobalMixInfluencesAdjustment) {
  HomeostasisConfig cfg;
  cfg.alpha = 1.0F;
  cfg.target_rate_hz = 5.0F;
  cfg.theta_step = 1.0F;
  cfg.global_mix = 1.0F;  // fully global
  cfg.theta_min = 0.1F;
  cfg.theta_max = 10.0F;

  Homeostasis homeo(cfg);
  float dt_ms = 1.0F;

  // Local r_avg is low, but global activity is high -> should increase theta.
  std::vector<float> theta = {1.0F};
  std::vector<float> r_avg = {0.0F};
  float global_activity = 1.0F;  // 1000 Hz equivalent at dt=1 ms

  auto theta_new = homeo.ComputeTheta(theta, r_avg, dt_ms, global_activity);
  EXPECT_GT(theta_new[0], theta[0]);

  // With global_mix = 0, same inputs should decrease theta (target above
  // local).
  cfg.global_mix = 0.0F;
  homeo.SetConfig(cfg);
  theta_new = homeo.ComputeTheta(theta, r_avg, dt_ms, global_activity);
  EXPECT_LT(theta_new[0], theta[0]);
}

}  // namespace senna::plasticity

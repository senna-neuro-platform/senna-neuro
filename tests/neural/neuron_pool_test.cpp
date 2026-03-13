#include "core/neural/neuron_pool.hpp"

#include <gtest/gtest.h>

#include "core/plasticity/homeostasis.hpp"
#include "core/spatial/lattice.hpp"

namespace senna::neural {
namespace {

class NeuronPoolTest : public ::testing::Test {
 protected:
  spatial::Lattice lattice_{2, 2, 1, 1.0, 42};
  LIFParams params_;
};

TEST_F(NeuronPoolTest, DecayWithoutInput) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.V(id) = 1.0F;
  pool.t_last(id) = 0.0F;
  pool.t_spike(id) = -100.0F;  // not refractory

  bool fired = pool.ReceiveInput(id, 10.0F, 0.0F);
  EXPECT_FALSE(fired);
  float expected = params_.V_rest +
                   (1.0F - params_.V_rest) * std::exp(-10.0F / params_.tau_m);
  EXPECT_NEAR(pool.V(id), expected, 1e-4F);
}

TEST_F(NeuronPoolTest, RefractoryBlocksInput) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.V(id) = 0.5F;
  pool.t_spike(id) = 9.5F;  // refractory for t_ref=2 at t=10
  bool fired = pool.ReceiveInput(id, 10.0F, 1.0F);
  EXPECT_FALSE(fired);
  EXPECT_FLOAT_EQ(pool.V(id), 0.5F);
}

TEST_F(NeuronPoolTest, FireResetsState) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.Fire(id, 5.0F);
  EXPECT_FLOAT_EQ(pool.V(id), params_.V_reset);
  EXPECT_FLOAT_EQ(pool.t_spike(id), 5.0F);
  EXPECT_FLOAT_EQ(pool.t_last(id), 5.0F);
}

TEST_F(NeuronPoolTest, ThresholdTriggersSpike) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.t_spike(id) = -100.0F;
  bool fired = pool.ReceiveInput(id, 0.5F, params_.theta_base + 0.2F);
  EXPECT_TRUE(fired);
  pool.Fire(id, 0.5F);
  EXPECT_FLOAT_EQ(pool.V(id), params_.V_reset);
}

TEST_F(NeuronPoolTest, HomeostasisAdjustsThetaUpAndDown) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  // One neuron fires repeatedly, another stays quiet.
  std::vector<int32_t> fired = {0};
  plasticity::HomeostasisConfig cfg;
  cfg.alpha = 0.5F;
  cfg.target_rate_hz = 10.0F;
  cfg.theta_step = 0.01F;
  plasticity::Homeostasis homeo(cfg);
  for (int i = 0; i < 5; ++i) {
    pool.UpdateAverages(fired, cfg.alpha);
    auto theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(),
                                        pool.RateSnapshot(), /*dt_ms=*/1.0F,
                                        /*global_activity=*/-1.0F);
    pool.ApplyThetaBuffer(theta_new);
  }
  EXPECT_GT(pool.theta(0), params_.theta_base);
  EXPECT_LT(pool.theta(1), params_.theta_base);
}

TEST_F(NeuronPoolTest, HomeostasisParametersAffectRate) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> fired = {0};
  plasticity::HomeostasisConfig cfg;
  cfg.alpha = 0.1F;
  cfg.target_rate_hz = 5.0F;  // push silent neurons to become more excitable
  cfg.theta_step = 0.5F;
  plasticity::Homeostasis homeo(cfg);
  pool.UpdateAverages(fired, cfg.alpha);
  auto theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                      /*dt_ms=*/1.0F,
                                      /*global_activity=*/-1.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_GT(pool.theta(0), params_.theta_base + 1.0F);
  // Quiet neuron should decrease threshold noticeably.
  EXPECT_LT(pool.theta(1), params_.theta_base - 0.1F);
}

TEST_F(NeuronPoolTest, HomeostasisUsesGlobalActivitySignal) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> none;
  float theta0 = pool.theta(0);
  plasticity::HomeostasisConfig cfg;
  cfg.alpha = 0.9F;
  cfg.target_rate_hz = 10.0F;
  cfg.theta_step = 0.2F;
  plasticity::Homeostasis homeo(cfg);

  // Global activity above target raises thresholds even if neuron was silent.
  pool.UpdateAverages(none, cfg.alpha);
  auto theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                      /*dt_ms=*/1.0F,
                                      /*global_activity=*/0.5F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_GT(pool.theta(0), theta0);

  // Global activity below target lowers thresholds.
  theta0 = pool.theta(0);
  cfg.target_rate_hz = 0.6F;
  homeo.SetConfig(cfg);
  pool.UpdateAverages(none, cfg.alpha);
  theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                 /*dt_ms=*/1.0F, /*global_activity=*/0.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_LT(pool.theta(0), theta0);
}

TEST_F(NeuronPoolTest, HomeostasisAlphaControlsAveraging) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> fired = {0};
  plasticity::HomeostasisConfig cfg;
  cfg.alpha = 0.5F;
  cfg.target_rate_hz = 0.5F;
  cfg.theta_step = 0.0F;  // freeze threshold to isolate r_avg update
  plasticity::Homeostasis homeo(cfg);

  pool.UpdateAverages(fired, cfg.alpha);
  auto theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                      /*dt_ms=*/1.0F,
                                      /*global_activity=*/-1.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_NEAR(pool.r_avg(0), 0.5F, 1e-5F);
  pool.UpdateAverages(fired, cfg.alpha);
  theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                 /*dt_ms=*/1.0F, /*global_activity=*/-1.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_NEAR(pool.r_avg(0), 0.75F, 1e-5F);
}

TEST_F(NeuronPoolTest, HomeostasisClampsThetaBounds) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  plasticity::HomeostasisConfig cfg;
  cfg.alpha = 0.9F;
  cfg.target_rate_hz = 0.0F;
  cfg.theta_step = 1.0F;
  cfg.theta_min = 0.8F;
  cfg.theta_max = 1.2F;
  plasticity::Homeostasis homeo(cfg);

  // Force large increase but clamp to theta_max.
  std::vector<int32_t> fired = {0};
  pool.UpdateAverages(fired, cfg.alpha);
  auto theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                      /*dt_ms=*/1.0F,
                                      /*global_activity=*/-1.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_LE(pool.theta(0), cfg.theta_max);

  // Force large decrease but clamp to theta_min.
  cfg.target_rate_hz = 1000.0F;  // extremely high target, neuron silent
  homeo.SetConfig(cfg);
  std::vector<int32_t> none;
  pool.UpdateAverages(none, cfg.alpha);
  theta_new = homeo.ComputeTheta(pool.ThetaSnapshot(), pool.RateSnapshot(),
                                 /*dt_ms=*/1.0F, /*global_activity=*/-1.0F);
  pool.ApplyThetaBuffer(theta_new);
  EXPECT_GE(pool.theta(1), cfg.theta_min);
}

TEST_F(NeuronPoolTest, ExcitatoryRatioRespected) {
  double ratio = 0.25;
  NeuronPool pool(lattice_, params_, ratio, 999);
  int exc = 0;
  for (int i = 0; i < pool.size(); ++i) {
    if (pool.type(i) == NeuronType::Excitatory) ++exc;
  }
  double observed = static_cast<double>(exc) / pool.size();
  EXPECT_NEAR(observed, ratio, 0.25);  // allow variance for small N
}

}  // namespace
}  // namespace senna::neural

#include "core/neural/neuron_pool.hpp"

#include <gtest/gtest.h>

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
  pool.V(id) = 1.0f;
  pool.t_last(id) = 0.0f;
  pool.t_spike(id) = -100.0f;  // not refractory

  bool fired = pool.ReceiveInput(id, 10.0f, 0.0f);
  EXPECT_FALSE(fired);
  float expected = params_.V_rest +
                   (1.0f - params_.V_rest) * std::exp(-10.0f / params_.tau_m);
  EXPECT_NEAR(pool.V(id), expected, 1e-4f);
}

TEST_F(NeuronPoolTest, RefractoryBlocksInput) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.V(id) = 0.5f;
  pool.t_spike(id) = 9.5f;  // refractory for t_ref=2 at t=10
  bool fired = pool.ReceiveInput(id, 10.0f, 1.0f);
  EXPECT_FALSE(fired);
  EXPECT_FLOAT_EQ(pool.V(id), 0.5f);
}

TEST_F(NeuronPoolTest, FireResetsState) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.Fire(id, 5.0f);
  EXPECT_FLOAT_EQ(pool.V(id), params_.V_reset);
  EXPECT_FLOAT_EQ(pool.t_spike(id), 5.0f);
  EXPECT_FLOAT_EQ(pool.t_last(id), 5.0f);
}

TEST_F(NeuronPoolTest, ThresholdTriggersSpike) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  int id = 0;
  pool.t_spike(id) = -100.0f;
  bool fired = pool.ReceiveInput(id, 0.5f, params_.theta_base + 0.2f);
  EXPECT_TRUE(fired);
  pool.Fire(id, 0.5f);
  EXPECT_FLOAT_EQ(pool.V(id), params_.V_reset);
}

TEST_F(NeuronPoolTest, HomeostasisAdjustsThetaUpAndDown) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  // One neuron fires repeatedly, another stays quiet.
  std::vector<int32_t> fired = {0};
  for (int i = 0; i < 5; ++i) {
    pool.ApplyHomeostasis(fired, 0.5f, 0.1f, 0.05f);
  }
  EXPECT_GT(pool.theta(0), params_.theta_base);
  EXPECT_LT(pool.theta(1), params_.theta_base);
}

TEST_F(NeuronPoolTest, HomeostasisParametersAffectRate) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> fired = {0};
  pool.ApplyHomeostasis(fired, 0.1f, 0.0f, 0.5f);  // aggressive increase
  EXPECT_GT(pool.theta(0), params_.theta_base + 0.1f);
  // Quiet neuron should decrease threshold noticeably.
  EXPECT_LT(pool.theta(1), params_.theta_base - 0.05f);
}

TEST_F(NeuronPoolTest, HomeostasisUsesGlobalActivitySignal) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> none;
  float theta0 = pool.theta(0);

  // Global activity above target raises thresholds even if neuron was silent.
  pool.ApplyHomeostasis(none, 0.9f, 0.1f, 0.2f, /*global_activity=*/0.5f);
  EXPECT_GT(pool.theta(0), theta0);

  // Global activity below target lowers thresholds.
  theta0 = pool.theta(0);
  pool.ApplyHomeostasis(none, 0.9f, 0.6f, 0.2f, /*global_activity=*/0.0f);
  EXPECT_LT(pool.theta(0), theta0);
}

TEST_F(NeuronPoolTest, HomeostasisAlphaControlsAveraging) {
  NeuronPool pool(lattice_, params_, 1.0, 123);
  std::vector<int32_t> fired = {0};

  pool.ApplyHomeostasis(fired, 0.5f, 0.5f, 0.0f);
  EXPECT_NEAR(pool.r_avg(0), 0.5f, 1e-5f);
  pool.ApplyHomeostasis(fired, 0.5f, 0.5f, 0.0f);
  EXPECT_NEAR(pool.r_avg(0), 0.75f, 1e-5f);
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

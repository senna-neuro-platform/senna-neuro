#include "core/neural/neuron_pool.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace senna::neural {
namespace {

class NeuronPoolTest : public ::testing::Test {
 protected:
  static constexpr int kN = 100;
  static constexpr double kExcRatio = 0.8;
  static constexpr uint64_t kSeed = 42;

  spatial::Lattice lattice_{10, 10, 1, 1.0, kSeed};
  NeuronPool pool_{lattice_, kDefaultLIF, kExcRatio, kSeed};
};

// --- 2.1 State initialization ---

TEST_F(NeuronPoolTest, SizeMatchesLattice) {
  EXPECT_EQ(pool_.size(), lattice_.neuron_count());
}

TEST_F(NeuronPoolTest, InitialVoltageIsVrest) {
  for (int i = 0; i < pool_.size(); ++i) {
    EXPECT_FLOAT_EQ(pool_.V(i), kDefaultLIF.V_rest);
  }
}

TEST_F(NeuronPoolTest, InitialThresholdIsThetaBase) {
  for (int i = 0; i < pool_.size(); ++i) {
    EXPECT_FLOAT_EQ(pool_.theta(i), kDefaultLIF.theta_base);
  }
}

TEST_F(NeuronPoolTest, ExcitatoryInhibitoryRatio) {
  int exc = 0;
  for (int i = 0; i < pool_.size(); ++i) {
    if (pool_.type(i) == NeuronType::Excitatory) ++exc;
  }
  double ratio = static_cast<double>(exc) / pool_.size();
  EXPECT_GE(ratio, 0.65);
  EXPECT_LE(ratio, 0.95);
}

TEST_F(NeuronPoolTest, SignMatchesType) {
  for (int i = 0; i < pool_.size(); ++i) {
    if (pool_.type(i) == NeuronType::Excitatory) {
      EXPECT_FLOAT_EQ(pool_.sign(i), 1.0f);
    } else {
      EXPECT_FLOAT_EQ(pool_.sign(i), -1.0f);
    }
  }
}

TEST_F(NeuronPoolTest, NotRefractoryAtStart) {
  // t_spike is initialized to -t_ref, so at t=0 neuron is NOT refractory.
  for (int i = 0; i < pool_.size(); ++i) {
    EXPECT_FALSE(pool_.IsRefractory(i, 0.0f));
  }
}

// --- 2.2 Lazy decay ---

TEST_F(NeuronPoolTest, DecayAfterOneTau) {
  int id = 0;
  pool_.V(id) = 1.0f;
  pool_.t_last(id) = 0.0f;

  // After tau_m ms, V should decay to ~0.368 (1/e).
  pool_.ReceiveInput(id, kDefaultLIF.tau_m, 0.0f);
  EXPECT_NEAR(pool_.V(id), std::exp(-1.0f), 1e-5f);
}

TEST_F(NeuronPoolTest, NoDecayIfNoTimePassed) {
  int id = 0;
  pool_.V(id) = 0.5f;
  pool_.t_last(id) = 10.0f;

  pool_.ReceiveInput(id, 10.0f, 0.0f);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.5f);
}

TEST_F(NeuronPoolTest, DecayPlusInput) {
  int id = 0;
  pool_.V(id) = 1.0f;
  pool_.t_last(id) = 0.0f;

  // After tau_m ms: V decays to e^-1 ~ 0.368, then add 0.5.
  pool_.ReceiveInput(id, kDefaultLIF.tau_m, 0.5f);
  float expected = std::exp(-1.0f) + 0.5f;
  EXPECT_NEAR(pool_.V(id), expected, 1e-5f);
}

TEST_F(NeuronPoolTest, DecayTowardsVrest) {
  int id = 0;
  pool_.V(id) = 0.8f;
  pool_.t_last(id) = 0.0f;

  // After a very long time, V should approach V_rest (0.0).
  pool_.ReceiveInput(id, 1000.0f, 0.0f);
  EXPECT_NEAR(pool_.V(id), kDefaultLIF.V_rest, 1e-5f);
}

// --- 2.3 Spike generation ---

TEST_F(NeuronPoolTest, SpikeOnThresholdCrossing) {
  int id = 0;
  pool_.V(id) = 0.9f;
  pool_.t_last(id) = 10.0f;

  // Input of 0.2 at same time: V = 0.9 + 0.2 = 1.1 >= theta=1.0.
  bool fired = pool_.ReceiveInput(id, 10.0f, 0.2f);
  EXPECT_TRUE(fired);
}

TEST_F(NeuronPoolTest, NoSpikeUnderThreshold) {
  int id = 0;
  pool_.V(id) = 0.5f;
  pool_.t_last(id) = 10.0f;

  bool fired = pool_.ReceiveInput(id, 10.0f, 0.1f);
  EXPECT_FALSE(fired);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.6f);
}

TEST_F(NeuronPoolTest, FireResetsVoltage) {
  int id = 0;
  pool_.V(id) = 1.5f;
  pool_.Fire(id, 10.0f);

  EXPECT_FLOAT_EQ(pool_.V(id), kDefaultLIF.V_reset);
  EXPECT_FLOAT_EQ(pool_.t_spike(id), 10.0f);
}

// --- 2.3 Refractory period ---

TEST_F(NeuronPoolTest, RefractoryIgnoresInput) {
  int id = 0;
  pool_.Fire(id, 10.0f);

  // During refractory (t < t_spike + t_ref = 12.0).
  bool fired = pool_.ReceiveInput(id, 11.0f, 100.0f);
  EXPECT_FALSE(fired);
  EXPECT_FLOAT_EQ(pool_.V(id), kDefaultLIF.V_reset);
}

TEST_F(NeuronPoolTest, AcceptsInputAfterRefractory) {
  int id = 0;
  pool_.Fire(id, 10.0f);

  // After refractory period ends (t >= t_spike + t_ref = 12.0).
  bool fired = pool_.ReceiveInput(id, 12.0f, 0.5f);
  EXPECT_FALSE(fired);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.5f);
}

TEST_F(NeuronPoolTest, RefractoryBoundary) {
  int id = 0;
  pool_.Fire(id, 10.0f);

  // Exactly at boundary: t = 10.0 + 2.0 = 12.0 — NOT refractory.
  EXPECT_FALSE(pool_.IsRefractory(id, 12.0f));
  // Just before: t = 11.99 — still refractory.
  EXPECT_TRUE(pool_.IsRefractory(id, 11.99f));
}

// --- 2.4 E/I types ---

TEST_F(NeuronPoolTest, ExcitatoryPositiveOutput) {
  // Find an excitatory neuron.
  for (int i = 0; i < pool_.size(); ++i) {
    if (pool_.type(i) == NeuronType::Excitatory) {
      EXPECT_FLOAT_EQ(pool_.sign(i), 1.0f);
      return;
    }
  }
  FAIL() << "No excitatory neuron found";
}

TEST_F(NeuronPoolTest, InhibitoryNegativeOutput) {
  for (int i = 0; i < pool_.size(); ++i) {
    if (pool_.type(i) == NeuronType::Inhibitory) {
      EXPECT_FLOAT_EQ(pool_.sign(i), -1.0f);
      return;
    }
  }
  FAIL() << "No inhibitory neuron found";
}

// --- SoA correctness ---

TEST_F(NeuronPoolTest, GetSetRoundTrip) {
  int id = 5;
  pool_.V(id) = 0.42f;
  pool_.theta(id) = 1.5f;
  pool_.t_last(id) = 3.0f;
  pool_.t_spike(id) = 1.0f;
  pool_.r_avg(id) = 7.0f;

  Neuron n = pool_.Get(id, lattice_);
  EXPECT_FLOAT_EQ(n.V, 0.42f);
  EXPECT_FLOAT_EQ(n.theta, 1.5f);
  EXPECT_FLOAT_EQ(n.t_last, 3.0f);
  EXPECT_FLOAT_EQ(n.t_spike, 1.0f);
  EXPECT_FLOAT_EQ(n.r_avg, 7.0f);

  n.V = 0.99f;
  n.theta = 2.0f;
  pool_.Set(id, n);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.99f);
  EXPECT_FLOAT_EQ(pool_.theta(id), 2.0f);
}

// --- Full spike cycle ---

TEST_F(NeuronPoolTest, FullSpikeCycle) {
  int id = 0;

  // 1. Accumulate input to cross threshold.
  pool_.ReceiveInput(id, 0.0f, 0.6f);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.6f);

  bool fired = pool_.ReceiveInput(id, 0.0f, 0.5f);
  EXPECT_TRUE(fired);
  EXPECT_FLOAT_EQ(pool_.V(id), 1.1f);

  // 2. Fire.
  pool_.Fire(id, 0.0f);
  EXPECT_FLOAT_EQ(pool_.V(id), 0.0f);

  // 3. Refractory — input ignored.
  fired = pool_.ReceiveInput(id, 1.0f, 10.0f);
  EXPECT_FALSE(fired);

  // 4. After refractory, neuron accepts input again.
  fired = pool_.ReceiveInput(id, 2.0f, 0.3f);
  EXPECT_FALSE(fired);
  EXPECT_NEAR(pool_.V(id), 0.3f, 1e-5f);
}

}  // namespace
}  // namespace senna::neural

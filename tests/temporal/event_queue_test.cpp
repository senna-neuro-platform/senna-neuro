#include "core/temporal/event_queue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <thread>
#include <vector>

#include "core/temporal/time_manager.hpp"

namespace senna::temporal {
namespace {

// --- 4.1 EventQueue ---

TEST(EventQueueTest, EmptyOnConstruction) {
  EventQueue q;
  EXPECT_TRUE(q.empty());
  EXPECT_EQ(q.size(), 0);
  EXPECT_FLOAT_EQ(q.PeekTime(), -1.0f);
}

TEST(EventQueueTest, PushAndPeek) {
  EventQueue q;
  q.Push({.target_id = 0, .source_id = 1, .arrival_time = 5.0f, .value = 0.1f});
  q.Push({.target_id = 1, .source_id = 2, .arrival_time = 3.0f, .value = 0.2f});

  EXPECT_EQ(q.size(), 2);
  EXPECT_FLOAT_EQ(q.PeekTime(), 3.0f);
}

TEST(EventQueueTest, DrainInTimeOrder) {
  EventQueue q;
  q.Push({.target_id = 0, .source_id = 0, .arrival_time = 5.0f, .value = 0.1f});
  q.Push({.target_id = 1, .source_id = 0, .arrival_time = 1.0f, .value = 0.2f});
  q.Push({.target_id = 2, .source_id = 0, .arrival_time = 3.0f, .value = 0.3f});

  std::vector<SpikeEvent> out;
  int count = q.DrainUntil(10.0f, out);

  EXPECT_EQ(count, 3);
  EXPECT_FLOAT_EQ(out[0].arrival_time, 1.0f);
  EXPECT_FLOAT_EQ(out[1].arrival_time, 3.0f);
  EXPECT_FLOAT_EQ(out[2].arrival_time, 5.0f);
  EXPECT_TRUE(q.empty());
}

TEST(EventQueueTest, DrainRespectsTimeWindow) {
  EventQueue q;
  q.Push({.target_id = 0, .source_id = 0, .arrival_time = 1.0f, .value = 0.1f});
  q.Push({.target_id = 1, .source_id = 0, .arrival_time = 1.3f, .value = 0.2f});
  q.Push({.target_id = 2, .source_id = 0, .arrival_time = 1.5f, .value = 0.3f});
  q.Push({.target_id = 3, .source_id = 0, .arrival_time = 2.0f, .value = 0.4f});

  // Drain [0, 1.5) — should get events at 1.0 and 1.3.
  std::vector<SpikeEvent> out;
  int count = q.DrainUntil(1.5f, out);

  EXPECT_EQ(count, 2);
  EXPECT_FLOAT_EQ(out[0].arrival_time, 1.0f);
  EXPECT_FLOAT_EQ(out[1].arrival_time, 1.3f);
  EXPECT_EQ(q.size(), 2);  // 1.5 and 2.0 remain
}

TEST(EventQueueTest, QuantizationSameTick) {
  // Events at 1.1 and 1.3 with dt=0.5: tick [1.0, 1.5) captures both.
  EventQueue q;
  q.Push({.target_id = 0, .source_id = 0, .arrival_time = 1.1f, .value = 0.1f});
  q.Push({.target_id = 1, .source_id = 0, .arrival_time = 1.3f, .value = 0.2f});

  std::vector<SpikeEvent> out;
  q.DrainUntil(1.5f, out);
  EXPECT_EQ(out.size(), 2);
}

TEST(EventQueueTest, BoundaryEventStaysForNextTick) {
  EventQueue q;
  q.Push({.target_id = 0, .source_id = 0, .arrival_time = 1.5f, .value = 0.1f});
  std::vector<SpikeEvent> out;
  q.DrainUntil(1.5f, out);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(q.size(), 1u);
}

TEST(EventQueueTest, EmptyDrain) {
  EventQueue q;
  std::vector<SpikeEvent> out;
  int count = q.DrainUntil(10.0f, out);
  EXPECT_EQ(count, 0);
  EXPECT_TRUE(out.empty());
}

TEST(EventQueueTest, PushBatch) {
  EventQueue q;
  std::vector<SpikeEvent> batch = {
      {.target_id = 0, .source_id = 0, .arrival_time = 3.0f, .value = 0.1f},
      {.target_id = 1, .source_id = 0, .arrival_time = 1.0f, .value = 0.2f},
      {.target_id = 2, .source_id = 0, .arrival_time = 2.0f, .value = 0.3f},
  };
  q.PushBatch(batch);
  EXPECT_EQ(q.size(), 3);
  EXPECT_FLOAT_EQ(q.PeekTime(), 1.0f);
}

TEST(EventQueueTest, ConcurrentPush) {
  EventQueue q;
  constexpr int kPerThread = 1000;
  constexpr int kThreads = 4;

  auto pusher = [&q](int thread_id) {
    for (int i = 0; i < kPerThread; ++i) {
      q.Push({.target_id = thread_id,
              .source_id = 0,
              .arrival_time = static_cast<float>(i),
              .value = 0.1f});
    }
  };

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(pusher, t);
  }
  for (auto& t : threads) t.join();

  EXPECT_EQ(q.size(), kPerThread * kThreads);

  // Drain all and verify count.
  std::vector<SpikeEvent> out;
  q.DrainUntil(static_cast<float>(kPerThread + 1), out);
  EXPECT_EQ(out.size(), kPerThread * kThreads);
}

// --- 4.3 TimeManager with chain ---

class TimeManagerTest : public ::testing::Test {
 protected:
  static constexpr uint64_t kSeed = 42;

  // Tiny lattice: 3x3x1, density 1.0, radius 1.5 (direct neighbors only).
  spatial::Lattice lattice_{3, 3, 1, 1.0, kSeed};
  spatial::NeighborIndex neighbors_{lattice_, 1.5f, 1};
  neural::NeuronPool pool_{lattice_, neural::kDefaultLIF, 0.8, kSeed};
  synaptic::SynapseIndex synapses_{lattice_, neighbors_, pool_};
};

TEST_F(TimeManagerTest, EmptyTickAdvancesTime) {
  EventQueue queue;
  TimeManager tm(0.5f);

  auto fired = tm.Tick(queue, pool_, synapses_);
  EXPECT_TRUE(fired.empty());
  EXPECT_FLOAT_EQ(tm.time(), 0.5f);
}

TEST_F(TimeManagerTest, EventDelivered) {
  EventQueue queue;
  TimeManager tm(0.5f);

  // Inject an event that arrives at t=0.1 with a large value.
  queue.Push(
      {.target_id = 0, .source_id = 1, .arrival_time = 0.1f, .value = 0.5f});

  tm.Tick(queue, pool_, synapses_);

  // Neuron 0 should have received the input.
  // V starts at 0 (V_rest), input 0.5 -> V = 0.5 (below theta=1.0).
  // But it may have fired if there were subsequent events. Just check V
  // changed. Since 0.5 < 1.0, it shouldn't fire.
  EXPECT_NEAR(pool_.V(0), 0.5f, 1e-4f);
}

TEST_F(TimeManagerTest, SpikeGeneratesNewEvents) {
  EventQueue queue;
  TimeManager tm(0.5f);

  // Push event that makes neuron 0 fire (value >= theta).
  queue.Push(
      {.target_id = 0, .source_id = 1, .arrival_time = 0.1f, .value = 1.5f});

  auto fired = tm.Tick(queue, pool_, synapses_);

  EXPECT_FALSE(fired.empty());
  EXPECT_EQ(fired[0], 0);

  // New events should be in the queue (from outgoing synapses of neuron 0).
  EXPECT_FALSE(queue.empty());
  EXPECT_GT(queue.size(), 0);
}

TEST_F(TimeManagerTest, ChainPropagation) {
  EventQueue queue;
  TimeManager tm(0.5f);

  // Make neuron 0 fire by injecting strong input.
  queue.Push(
      {.target_id = 0, .source_id = -1, .arrival_time = 0.1f, .value = 1.5f});

  // Run enough ticks for events to propagate through the network.
  int total_spikes = 0;
  for (int i = 0; i < 20; ++i) {
    auto fired = tm.Tick(queue, pool_, synapses_);
    total_spikes += static_cast<int>(fired.size());
  }

  // At least neuron 0 should have fired.
  EXPECT_GE(total_spikes, 1);
  // Time should have advanced by 20 * 0.5 = 10.0 ms.
  EXPECT_FLOAT_EQ(tm.time(), 10.0f);
}

}  // namespace
}  // namespace senna::temporal

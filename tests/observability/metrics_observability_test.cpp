#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <string>
#include <thread>
#include <vector>

#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"
#include "core/observability/prometheus_exporter.hpp"

using senna::observability::AggregatedMetrics;
using senna::observability::MetricsCollector;
using senna::observability::ObservabilityThread;
using senna::observability::PrometheusExporter;

TEST(MetricsCollectorTest, DropsWhenFullAndPreservesOrder) {
  MetricsCollector mc(/*capacity=*/2);
  EXPECT_TRUE(mc.RecordTickSummary(1, 10, 5, 0.5));
  EXPECT_TRUE(mc.RecordPrune(1));
  // buffer now full; next push drops
  EXPECT_FALSE(mc.RecordSprout(1));
  EXPECT_EQ(mc.Dropped(), 1U);

  auto e1 = mc.Pop();
  ASSERT_TRUE(e1);
  EXPECT_EQ(std::get<MetricsCollector::TickSummary>(e1->payload).tick_id, 1U);
  auto e2 = mc.Pop();
  ASSERT_TRUE(e2);
  EXPECT_EQ(std::get<MetricsCollector::PruneEvent>(e2->payload).count, 1U);
  EXPECT_FALSE(mc.Pop());
}

TEST(ObservabilityThreadTest, AggregatesBasicMetrics) {
  MetricsCollector mc;
  ObservabilityThread obs(mc, /*total_neurons=*/100);
  obs.Start();

  mc.RecordTickSummary(7, 10, 5, 1.0);
  mc.RecordPrune(2);
  mc.RecordSprout(3);
  mc.RecordWeightUpdate(0.1F, 0.5F);

  // allow worker to drain
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  AggregatedMetrics snap = obs.Snapshot();

  EXPECT_EQ(snap.last_tick_id, 7U);
  EXPECT_NEAR(snap.active_ratio, 0.10, 1e-3);
  EXPECT_EQ(snap.pruned_total, 2U);
  EXPECT_EQ(snap.sprouted_total, 3U);
  EXPECT_FLOAT_EQ(snap.last_weight, 0.5F);
  obs.Stop();
}

// Simple helper to connect to exporter and read all data.
static std::string FetchAll(const std::string& host, int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = inet_addr(host.c_str());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    return {};
  }
  const char req[] = "GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n";
  (void)send(fd, req, sizeof(req) - 1, 0);
  std::string out;
  std::array<char, 1024> buf{};
  ssize_t n = 0;
  while ((n = recv(fd, buf.data(), buf.size(), 0)) > 0) {
    out.append(buf.data(), static_cast<std::size_t>(n));
  }
  ::close(fd);
  return out;
}

TEST(PrometheusExporterTest, RespondsWithRenderedBody) {
  PrometheusExporter exp;
  auto render = []() { return "metric_test 1\n"; };
  ASSERT_TRUE(exp.Start(0, render));
  int port = exp.port();
  ASSERT_GT(port, 0);

  // Give the server a moment to start.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  std::string resp = FetchAll("127.0.0.1", port);
  exp.Stop();

  EXPECT_NE(resp.find("HTTP/1.1 200 OK"), std::string::npos);
  EXPECT_NE(resp.find("metric_test 1"), std::string::npos);
}

TEST(PrometheusRenderTest, FormatsRequiredMetrics) {
  AggregatedMetrics agg{};
  agg.active_ratio = 0.2;
  agg.spikes_per_tick = 5;
  agg.excitatory_rate_hz = 12.3;
  agg.inhibitory_rate_hz = 3.1;
  agg.ei_balance = 3.97;
  agg.train_accuracy = 0.8;
  agg.test_accuracy = 0.7;
  agg.synapse_count = 1234;
  agg.pruned_total = 10;
  agg.sprouted_total = 20;
  agg.ticks_per_second = 150.5;
  agg.memory_bytes = 1024;
  agg.phase = 0.0;
  agg.sleep_pressure = 0.5;
  agg.virtual_time_ms = 42;
  agg.tick_duration_buckets = {0.001};
  agg.tick_duration_counts = {1, 0};
  agg.tick_duration_count = 1;
  agg.tick_duration_sum = 0.0008;

  auto text = ObservabilityThread::RenderPrometheus(agg);
  EXPECT_NE(text.find("senna_active_neurons_ratio"), std::string::npos);
  EXPECT_NE(text.find("senna_tick_duration_seconds_bucket"), std::string::npos);
  EXPECT_NE(text.find("senna_tick_duration_seconds_sum"), std::string::npos);
}

TEST(PrometheusLivePipelineTest, MetricsUpdateWhileRunning) {
  using namespace std::chrono_literals;
  MetricsCollector metrics;
  senna::network::NetworkConfig cfg;
  cfg.width = 6;
  cfg.height = 6;
  cfg.depth = 4;
  cfg.density = 0.6;
  cfg.dt = 1.0F;
  senna::network::Network net(cfg, &metrics);
  ObservabilityThread obs(metrics, net.lattice().neuron_count());
  obs.Start();
  PrometheusExporter exp;
  ASSERT_TRUE(exp.Start(0, senna::observability::MakePrometheusRender(obs)));
  int port = exp.port();
  ASSERT_GT(port, 0);

  for (int i = 0; i < 50; ++i) {
    auto syn = net.synapses_ptr();
    net.time_manager().Tick(net.queue(), net.pool(), *syn);
    std::this_thread::sleep_for(2ms);
  }

  auto body = FetchAll("127.0.0.1", port);
  exp.Stop();
  obs.Stop();

  EXPECT_NE(body.find("senna_active_neurons_ratio"), std::string::npos);
  EXPECT_NE(body.find("senna_tick_duration_seconds_count"), std::string::npos);
  EXPECT_NE(body.find("senna_ticks_per_second"), std::string::npos);
}

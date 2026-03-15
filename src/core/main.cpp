#include <atomic>
#include <csignal>
#include <iostream>
#include <thread>

#include "core/config/runtime_config.hpp"
#include "core/interfaces/grpc_server.hpp"
#include "core/interfaces/ws_server.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"
#include "core/observability/prometheus_exporter.hpp"

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> g_running{true};

void HandleSignal(int /*signo*/) { g_running.store(false); }
}  // namespace

int main() {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // Load runtime configuration (defaults if file missing).
  auto cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);
  senna::observability::ObservabilityThread obs(
      metrics, net.lattice().neuron_count(), std::chrono::milliseconds(2),
      cfg.observability.tick_duration_buckets);
  net.UpdatePhase(0.0, 0.0);
  net.UpdateAccuracy(0.0, 0.0);
  obs.Start();
  senna::observability::PrometheusExporter prometheus;
  prometheus.Start(cfg.ports.metrics,
                   senna::observability::MakePrometheusRender(obs),
                   cfg.observability.exporter_backlog);

  senna::interfaces::WsServer ws_server(cfg.ports.ws, &net);
  ws_server.Start();

  senna::interfaces::GrpcServer grpc_server(cfg.ports.grpc, &net, &metrics);
  grpc_server.Start();

  const auto loop_sleep = std::chrono::milliseconds(cfg.loop_sleep_ms);
  std::thread loop_thread([&]() {
    while (g_running.load()) {
      auto syn = net.synapses_ptr();
      net.time_manager().Tick(net.queue(), net.pool(), *syn);
      std::this_thread::sleep_for(loop_sleep);
    }
  });

  // Block until signal.
  while (g_running.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  grpc_server.Stop();
  g_running.store(false);
  if (loop_thread.joinable()) {
    loop_thread.join();
  }
  prometheus.Stop();
  obs.Stop();
  ws_server.Stop();

  return 0;
}

#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <thread>

#include "core/config/runtime_config.hpp"
#include "core/interfaces/ws_server.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"
#include "core/observability/prometheus_exporter.hpp"

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> g_running{true};

void HandleSignal(int signo) {  // NOLINT(misc-unused-parameters)
  g_running.store(false);
}

int CreateListener(int port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    std::perror("socket");
    return -1;
  }

  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    std::perror("setsockopt");
    ::close(fd);
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* addr_ptr = reinterpret_cast<sockaddr*>(&addr);
  if (bind(fd, addr_ptr, sizeof(addr)) < 0) {
    std::perror("bind");
    ::close(fd);
    return -1;
  }

  if (listen(fd, 16) < 0) {
    std::perror("listen");
    ::close(fd);
    return -1;
  }

  return fd;
}

void ServeLoop(int port, bool respond_http) {
  int listen_fd = CreateListener(port);
  if (listen_fd < 0) {
    return;
  }

  while (g_running.load()) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    int ready = select(listen_fd + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0) {
      continue;
    }

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* client_ptr = reinterpret_cast<sockaddr*>(&client_addr);
    int client_fd = accept(listen_fd, client_ptr, &client_len);
    if (client_fd < 0) {
      continue;
    }

    if (respond_http) {
      const char* response =
          "HTTP/1.1 200 OK\r\n"
          "Content-Type: text/plain\r\n"
          "Content-Length: 2\r\n"
          "Connection: close\r\n"
          "\r\n"
          "OK";
      (void)send(client_fd, response, std::strlen(response), 0);
    }

    ::close(client_fd);
  }

  ::close(listen_fd);
}
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
  // Initialize derived metrics that are fed from external subsystems.
  net.UpdatePhase(0.0, 0.0);
  net.UpdateAccuracy(0.0, 0.0);
  obs.Start();
  senna::observability::PrometheusExporter prometheus;
  prometheus.Start(cfg.ports.metrics,
                   senna::observability::MakePrometheusRender(obs),
                   cfg.observability.exporter_backlog);

  senna::interfaces::WsServer ws_server(cfg.ports.ws, &net);
  ws_server.Start();

  const auto loop_sleep = std::chrono::milliseconds(cfg.loop_sleep_ms);
  std::thread loop_thread([&]() {
    while (g_running.load()) {
      auto syn = net.synapses_ptr();
      net.time_manager().Tick(net.queue(), net.pool(), *syn);
      std::this_thread::sleep_for(loop_sleep);
    }
  });

  std::thread grpc_thread(ServeLoop, cfg.ports.grpc, false);

  grpc_thread.join();
  g_running.store(false);
  if (loop_thread.joinable()) {
    loop_thread.join();
  }
  prometheus.Stop();
  obs.Stop();
  ws_server.Stop();

  return 0;
}

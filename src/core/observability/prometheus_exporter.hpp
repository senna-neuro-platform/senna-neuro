/**
 * @file prometheus_exporter.hpp
 * @brief Minimal built-in HTTP exporter for Prometheus text format (Шаг 11.3).
 */

#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace senna::observability {

// Lightweight HTTP server that serves metrics at GET /metrics.
class PrometheusExporter {
 public:
  using RenderCallback = std::function<std::string()>;

  PrometheusExporter() = default;
  ~PrometheusExporter();

  // Starts the exporter on the given port. Returns false on bind/listen error.
  bool Start(int port, RenderCallback render_cb, int backlog = 8);
  void Stop();

  bool IsRunning() const { return running_.load(std::memory_order_relaxed); }
  int port() const { return port_; }

 private:
  void Run();
  bool SetupListenSocket(int port);
  void CloseListenSocket();
  void ServeClient(int client_fd) const;

  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  int port_{-1};
  int backlog_{8};
  std::thread worker_;
  RenderCallback render_cb_;
};

}  // namespace senna::observability

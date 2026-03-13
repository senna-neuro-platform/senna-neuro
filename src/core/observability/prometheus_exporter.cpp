/**
 * @file prometheus_exporter.cpp
 * @brief Minimal HTTP exporter that replies with Prometheus text format.
 */

#include "core/observability/prometheus_exporter.hpp"

#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <iostream>

namespace senna::observability {
PrometheusExporter::~PrometheusExporter() { Stop(); }

bool PrometheusExporter::Start(int port, RenderCallback render_cb,
                               int backlog) {
  if (running_.exchange(true)) {
    return true;
  }
  render_cb_ = std::move(render_cb);
  backlog_ = backlog;
  if (!SetupListenSocket(port)) {
    running_.store(false);
    return false;
  }
  worker_ = std::thread(&PrometheusExporter::Run, this);
  return true;
}

void PrometheusExporter::Stop() {
  if (!running_.exchange(false)) {
    return;
  }
  CloseListenSocket();
  if (worker_.joinable()) {
    worker_.join();
  }
}

bool PrometheusExporter::SetupListenSocket(int port) {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    std::perror("prometheus_exporter socket");
    return false;
  }

  int opt = 1;
  if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    std::perror("prometheus_exporter setsockopt");
    CloseListenSocket();
    return false;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* addr_ptr = reinterpret_cast<sockaddr*>(&addr);
  if (bind(listen_fd_, addr_ptr, sizeof(addr)) < 0) {
    std::perror("prometheus_exporter bind");
    CloseListenSocket();
    return false;
  }

  if (listen(listen_fd_, backlog_) < 0) {
    std::perror("prometheus_exporter listen");
    CloseListenSocket();
    return false;
  }
  // If port was 0 (ephemeral), fetch the actual port.
  socklen_t len = sizeof(addr);
  if (getsockname(listen_fd_, addr_ptr, &len) == 0) {
    port_ = ntohs(addr.sin_port);
  } else {
    port_ = port;
  }
  return true;
}

void PrometheusExporter::CloseListenSocket() {
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  port_ = -1;
}

void PrometheusExporter::Run() {
  while (running_.load(std::memory_order_relaxed)) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd_, &rfds);
    timeval tv{};
    tv.tv_sec = 1;
    tv.tv_usec = 0;

    int ready = select(listen_fd_ + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0) {
      continue;
    }

    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* client_ptr = reinterpret_cast<sockaddr*>(&client_addr);
    int client_fd = accept(listen_fd_, client_ptr, &client_len);
    if (client_fd < 0) {
      continue;
    }
    ServeClient(client_fd);
    ::close(client_fd);
  }
}

void PrometheusExporter::ServeClient(int client_fd) const {
  // best-effort read and ignore request
  std::array<char, 1024> buffer{};
  (void)read(client_fd, buffer.data(), buffer.size());

  std::string body = render_cb_ ? render_cb_() : "";
  std::string header =
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/plain; version=0.0.4\r\n"
      "Content-Length: " +
      std::to_string(body.size()) +
      "\r\n"
      "Connection: close\r\n"
      "\r\n";
  std::string response = header + body;
  (void)send(client_fd, response.data(), response.size(), 0);
}

}  // namespace senna::observability

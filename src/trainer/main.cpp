#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "core/config/runtime_config.hpp"

namespace {
bool TryConnect(const char* host, const char* port) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  addrinfo* result = nullptr;
  if (getaddrinfo(host, port, &hints, &result) != 0) {
    return false;
  }

  bool connected = false;
  for (addrinfo* rp = result; rp != nullptr; rp = rp->ai_next) {
    int fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) {
      continue;
    }

    if (connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      connected = true;
      close(fd);
      break;
    }

    close(fd);
  }

  freeaddrinfo(result);
  return connected;
}
}  // namespace

int main() {
  auto cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
  const char* host_env =
      std::getenv("SENNA_CORE_HOST");  // NOLINT(concurrency-mt-unsafe)
  const char* port_env =
      std::getenv("SENNA_CORE_PORT");  // NOLINT(concurrency-mt-unsafe)
  std::string host = host_env != nullptr ? host_env : cfg.trainer.host;
  std::string port =
      port_env != nullptr ? port_env : std::to_string(cfg.trainer.port);

  const int max_attempts = 30;
  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    if (TryConnect(host.c_str(), port.c_str())) {
      std::cout << "connected\n";
      break;
    }
    std::cout << "waiting for senna-core (" << attempt << "/" << max_attempts
              << ")\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }

  return 0;
}

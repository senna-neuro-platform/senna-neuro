#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

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
  const char* host = std::getenv("SENNA_CORE_HOST");
  const char* port = std::getenv("SENNA_CORE_PORT");
  if (host == nullptr) {
    host = "senna-core";
  }
  if (port == nullptr) {
    port = "50051";
  }

  const int max_attempts = 30;
  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    if (TryConnect(host, port)) {
      std::cout << "connected" << std::endl;
      break;
    }
    std::cout << "waiting for senna-core (" << attempt << "/" << max_attempts
              << ")" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  while (true) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }

  return 0;
}

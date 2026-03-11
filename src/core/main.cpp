#include <atomic>
#include <csignal>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
std::atomic<bool> g_running{true};

void HandleSignal(int) {
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

  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
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
    int client_fd = accept(listen_fd, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
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

  std::thread grpc_thread(ServeLoop, 50051, false);
  std::thread ws_thread(ServeLoop, 8080, false);
  std::thread metrics_thread(ServeLoop, 9090, true);

  grpc_thread.join();
  ws_thread.join();
  metrics_thread.join();

  return 0;
}

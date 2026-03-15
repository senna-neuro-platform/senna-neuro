#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <boost/asio.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>

#include "core/network/network_builder.hpp"

namespace senna::interfaces {

// Minimal WebSocket server that streams a snapshot of the network and
// lightweight tick updates. Designed to be non-blocking for the core loop.
class WsServer {
 public:
  WsServer(int port, network::Network* net);
  ~WsServer();

  void Start();
  void Stop();

 private:
  void Run();
  void HandleSession(const std::shared_ptr<void>& guard,
                     boost::asio::ip::tcp::socket socket);
  std::string BuildNetworkState() const;
  std::string BuildTickUpdate() const;
  void HandleCommand(std::string_view msg,
                     boost::beast::websocket::stream<boost::asio::ip::tcp::socket>& ws);
  void SendSynapseChunks(
      boost::beast::websocket::stream<boost::asio::ip::tcp::socket>& ws);
  std::vector<uint8_t> PackCborDeflate(uint8_t type,
                                       const boost::json::value& doc) const;
  void EncodeCbor(const boost::json::value& v,
                  std::vector<uint8_t>& out) const;
  void SnapshotNetwork();

  int port_;
  network::Network* net_;
  std::atomic<bool> stop_{false};
  std::atomic<bool> paused_{false};
  std::atomic<int> update_interval_ticks_{10};
  std::atomic<uint64_t> last_sent_tick_{0};
  std::atomic<int> current_label_{-1};
  std::thread server_thread_;
  std::thread snapshot_thread_;
  std::atomic<std::shared_ptr<std::vector<uint8_t>>> netstate_buf_{nullptr};
  std::atomic<std::shared_ptr<std::vector<std::vector<uint8_t>>>> syn_chunks_buf_{nullptr};
  std::chrono::milliseconds snapshot_period_{2000};
};

}  // namespace senna::interfaces

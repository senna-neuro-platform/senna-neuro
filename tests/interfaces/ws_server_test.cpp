#include "core/interfaces/ws_server.hpp"

#include <gtest/gtest.h>
#include <zlib.h>

#include <atomic>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/config/runtime_config.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"

using namespace std::chrono_literals;

namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;
using WsClient = websocket::stream<tcp::socket>;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::vector<uint8_t> Inflate(const std::vector<uint8_t>& data) {
  std::vector<uint8_t> out(4096);
  uLongf dest_len = out.size();
  int ret = Z_BUF_ERROR;
  while (ret == Z_BUF_ERROR) {
    ret = uncompress(out.data(), &dest_len, data.data(),
                     static_cast<uLong>(data.size()));
    if (ret == Z_BUF_ERROR) {
      out.resize(out.size() * 2);
      dest_len = out.size();
    }
  }
  out.resize(dest_len);
  return ret == Z_OK ? out : std::vector<uint8_t>{};
}

// Minimal CBOR decoder for types emitted by WsServer.
class CborReader {
 public:
  explicit CborReader(const std::vector<uint8_t>& buf) : buf_(&buf) {}
  boost::json::value Parse() { return ParseValue(); }

 private:
  const std::vector<uint8_t>* buf_;
  size_t pos_{0};

  bool has(size_t n) const {
    return buf_ != nullptr && pos_ + n <= buf_->size();
  }

  uint64_t ReadUInt(uint8_t ai) {
    if (ai < 24) return ai;
    if (ai == 24) return ReadBytes(1);
    if (ai == 25) return ReadBytes(2);
    if (ai == 26) return ReadBytes(4);
    if (ai == 27) return ReadBytes(8);
    return 0;
  }

  uint64_t ReadBytes(size_t n) {
    if (!has(n)) return 0;
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v = (v << 8) | (*buf_)[pos_++];
    return v;
  }

  boost::json::value ParseValue() {
    if (!has(1)) return nullptr;
    uint8_t b = (*buf_)[pos_++];
    uint8_t major = b >> 5;
    uint8_t ai = b & 0x1F;
    switch (major) {
      case 0: {
        uint64_t val = ReadUInt(ai);
        if (val <= static_cast<uint64_t>(INT64_MAX))
          return boost::json::value_from(static_cast<int64_t>(val));
        return boost::json::value_from(val);
      }
      case 1: {
        uint64_t n = ReadUInt(ai);
        return boost::json::value_from(
            static_cast<int64_t>(-1 - static_cast<int64_t>(n)));
      }
      case 3: {
        uint64_t n = ReadUInt(ai);
        if (!has(n)) return nullptr;
        std::string s(buf_->begin() + static_cast<long>(pos_),
                      buf_->begin() + static_cast<long>(pos_ + n));
        pos_ += n;
        return boost::json::value_from(s);
      }
      case 4: {
        uint64_t n = ReadUInt(ai);
        boost::json::array arr;
        arr.reserve(n);
        for (uint64_t i = 0; i < n && has(1); ++i) arr.push_back(ParseValue());
        return arr;
      }
      case 5: {
        uint64_t n = ReadUInt(ai);
        boost::json::object obj;
        for (uint64_t i = 0; i < n && has(1); ++i) {
          auto key = ParseValue();
          auto val = ParseValue();
          if (key.is_string()) obj[key.as_string()] = val;
        }
        return obj;
      }
      case 7: {
        if (ai == 20) return boost::json::value_from(false);
        if (ai == 21) return boost::json::value_from(true);
        if (ai == 22) return nullptr;
        if (ai == 25 && has(2)) {
          pos_ += 2;
          return nullptr;
        }
        if (ai == 26 && has(4)) {
          pos_ += 4;
          return nullptr;
        }
        if (ai == 27 && has(8)) {
          uint64_t bits = ReadBytes(8);
          double d = 0.0;
          std::memcpy(&d, &bits, sizeof(d));
          return boost::json::value_from(d);
        }
        return nullptr;
      }
      default:
        return nullptr;
    }
  }
};

struct Frame {
  uint8_t type{0};
  boost::json::value json;
  bool valid{false};
};

// Read one binary frame using async read + deadline timer.
// This avoids the "Operation canceled" bug with non-blocking sockets on
// Beast websocket streams.
Frame ReadFrame(WsClient& ws, boost::asio::io_context& ioc,
                std::chrono::milliseconds timeout = 2000ms) {
  beast::flat_buffer buffer;
  boost::system::error_code read_ec;
  bool read_done = false;

  boost::asio::steady_timer timer(ioc, timeout);
  timer.async_wait([&](boost::system::error_code ec) {
    if (!ec && !read_done) {
      boost::system::error_code cancel_ec;
      ws.next_layer().cancel(cancel_ec);
    }
  });

  ws.async_read(buffer, [&](boost::system::error_code ec, size_t /*bytes*/) {
    read_ec = ec;
    read_done = true;
    timer.cancel();
  });

  ioc.restart();
  ioc.run();

  Frame f{};
  if (read_ec || !read_done) return f;

  std::vector<uint8_t> data(buffer.size());
  boost::asio::buffer_copy(boost::asio::buffer(data), buffer.data());
  if (data.size() < 5u) return f;

  f.type = data[0];
  uint32_t len = (static_cast<uint32_t>(data[1]) << 24) |
                 (static_cast<uint32_t>(data[2]) << 16) |
                 (static_cast<uint32_t>(data[3]) << 8) | data[4];
  if (len + 5 != data.size()) return f;

  std::vector<uint8_t> payload(data.begin() + 5, data.end());
  auto inflated = Inflate(payload);
  CborReader decoder(inflated);
  f.json = decoder.Parse();
  f.valid = true;
  return f;
}

// Connect a WS client to localhost:port.
WsClient MakeClient(boost::asio::io_context& ioc, int port) {
  tcp::resolver resolver(ioc);
  auto results = resolver.resolve("127.0.0.1", std::to_string(port));
  WsClient ws(ioc);
  ws.next_layer().connect(*results.begin());
  ws.handshake("127.0.0.1", "/");
  return ws;
}

// Send a text command (uses string_view to avoid null terminator).
void SendCommand(WsClient& ws, std::string_view cmd) {
  ws.write(boost::asio::buffer(cmd.data(), cmd.size()));
}

}  // namespace

// ===== 13.3 Tests ==========================================================

// 13.3.1 WebSocket connects, NetworkState arrives.
TEST(WsServerTest, ConnectsAndReceivesNetworkState) {
  senna::config::RuntimeConfig cfg =
      senna::config::LoadRuntimeConfig("configs/default.yaml");
  cfg.network.width = 4;
  cfg.network.height = 4;
  cfg.network.depth = 3;
  cfg.network.density = 0.5;
  cfg.network.neighbor_radius = 1.0F;
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);

  constexpr int kPort = 19090;
  senna::interfaces::WsServer server(kPort, &net);
  server.Start();
  std::this_thread::sleep_for(200ms);

  boost::asio::io_context ioc;
  auto ws = MakeClient(ioc, kPort);

  Frame netstate = ReadFrame(ws, ioc);
  ASSERT_TRUE(netstate.valid) << "no frame received";
  ASSERT_EQ(netstate.type, 1);
  ASSERT_TRUE(netstate.json.is_object());

  auto& ns = netstate.json.as_object();
  EXPECT_EQ(ns.at("type").as_string(), "NetworkState");
  EXPECT_GT(ns.at("neuron_count").as_int64(), 0);
  EXPECT_TRUE(ns.contains("neurons"));
  EXPECT_TRUE(ns.contains("synapse_count"));

  boost::system::error_code ec;
  ws.close(websocket::close_code::normal, ec);
  server.Stop();
}

// 13.3.2 Neuron count in NetworkState matches grid size.
TEST(WsServerTest, NeuronCountMatchesGridSize) {
  constexpr int kW = 4, kH = 4, kD = 3;

  senna::config::RuntimeConfig cfg =
      senna::config::LoadRuntimeConfig("configs/default.yaml");
  cfg.network.width = kW;
  cfg.network.height = kH;
  cfg.network.depth = kD;
  cfg.network.density = 1.0;
  cfg.network.neighbor_radius = 1.0F;
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);

  constexpr int kPort = 19091;
  senna::interfaces::WsServer server(kPort, &net);
  server.Start();
  std::this_thread::sleep_for(200ms);

  boost::asio::io_context ioc;
  auto ws = MakeClient(ioc, kPort);

  Frame netstate = ReadFrame(ws, ioc);
  ASSERT_TRUE(netstate.valid);
  ASSERT_EQ(netstate.type, 1);

  auto& ns = netstate.json.as_object();
  int64_t neuron_count = ns.at("neuron_count").as_int64();

  // Network may include output neurons beyond the lattice grid.
  int64_t expected = static_cast<int64_t>(net.pool().size());
  EXPECT_EQ(neuron_count, expected)
      << "WS neuron_count must match Network::pool().size()";

  // The grid itself must have at least W*H*D neurons (density=1).
  EXPECT_GE(neuron_count, kW * kH * kD);

  // neurons array size must match neuron_count.
  EXPECT_EQ(static_cast<int64_t>(ns.at("neurons").as_array().size()),
            neuron_count);

  boost::system::error_code ec;
  ws.close(websocket::close_code::normal, ec);
  server.Stop();
}

// 13.3.3 TickUpdate arrives with the specified interval.
TEST(WsServerTest, TickUpdateArrivesWithInterval) {
  senna::config::RuntimeConfig cfg =
      senna::config::LoadRuntimeConfig("configs/default.yaml");
  cfg.network.width = 3;
  cfg.network.height = 3;
  cfg.network.depth = 2;
  cfg.network.density = 0.6;
  cfg.network.neighbor_radius = 1.0F;
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);

  constexpr int kPort = 19092;
  senna::interfaces::WsServer server(kPort, &net);
  server.Start();
  std::this_thread::sleep_for(200ms);

  boost::asio::io_context ioc;
  auto ws = MakeClient(ioc, kPort);

  // Consume NetworkState (type 1) + all SynapseChunks (type 2).
  for (int i = 0; i < 20; ++i) {
    Frame f = ReadFrame(ws, ioc);
    if (!f.valid) break;
    // Stop after last SynapseChunk.
    if (f.type == 2 && f.json.is_object()) {
      auto& obj = f.json.as_object();
      if (obj.contains("last") && obj.at("last").as_bool()) break;
    }
  }

  // Request tick updates every 1 tick.
  SendCommand(ws, R"({"type":"SetUpdateInterval","ticks":1})");

  // Run ticks in background to generate snapshots.
  std::atomic<bool> running{true};
  std::thread tick_thread([&]() {
    for (int i = 0; i < 100 && running.load(); ++i) {
      auto syn = net.synapses_ptr();
      net.time_manager().Tick(net.queue(), net.pool(), *syn);
      std::this_thread::sleep_for(5ms);
    }
  });

  // Read frames looking for Ack (0x10) and TickUpdate (3).
  bool got_ack = false;
  bool got_tick = false;
  for (int i = 0; i < 30 && (!got_ack || !got_tick); ++i) {
    Frame f = ReadFrame(ws, ioc);
    if (!f.valid) continue;
    if (f.type == 0x10 && f.json.is_object()) {
      auto& ack = f.json.as_object();
      EXPECT_EQ(ack.at("cmd").as_string(), "SetUpdateInterval");
      EXPECT_EQ(ack.at("ok").as_bool(), true);
      got_ack = true;
    } else if (f.type == 3 && f.json.is_object()) {
      auto& tu = f.json.as_object();
      EXPECT_EQ(tu.at("type").as_string(), "TickUpdate");
      EXPECT_TRUE(tu.contains("t_ms"));
      EXPECT_TRUE(tu.contains("tick"));
      EXPECT_TRUE(tu.contains("fired"));
      got_tick = true;
    }
  }
  EXPECT_TRUE(got_ack) << "did not receive Ack for SetUpdateInterval";
  EXPECT_TRUE(got_tick) << "did not receive any TickUpdate";

  running.store(false);
  if (tick_thread.joinable()) tick_thread.join();

  boost::system::error_code ec;
  ws.close(websocket::close_code::normal, ec);
  server.Stop();
}

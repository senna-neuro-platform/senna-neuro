#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>
#include <gtest/gtest.h>
#include <zlib.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "core/config/runtime_config.hpp"
#include "core/interfaces/ws_server.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"

using namespace std::chrono_literals;

namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

namespace {

// Inflate zlib-compressed buffer.
std::vector<uint8_t> Inflate(const std::vector<uint8_t>& data) {
  std::vector<uint8_t> out(1024);
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
  if (ret != Z_OK) return {};
  return out;
}

// Minimal CBOR reader for types emitted by WsServer (maps/arrays/ints/strings/double/null).
class CborReader {
 public:
  explicit CborReader(const std::vector<uint8_t>& buf) : buf_(buf) {}

  boost::json::value Parse() { return ParseValue(); }

 private:
  const std::vector<uint8_t>& buf_;
  size_t pos_{0};

  bool has(size_t n) const { return pos_ + n <= buf_.size(); }

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
    for (size_t i = 0; i < n; ++i) {
      v = (v << 8) | buf_[pos_++];
    }
    return v;
  }

  boost::json::value ParseValue() {
    if (!has(1)) return nullptr;
    uint8_t b = buf_[pos_++];
    uint8_t major = b >> 5;
    uint8_t ai = b & 0x1F;
    switch (major) {
      case 0: {  // positive int
        return boost::json::value_from(ReadUInt(ai));
      }
      case 1: {  // negative int
        uint64_t n = ReadUInt(ai);
        int64_t val = -1 - static_cast<int64_t>(n);
        return boost::json::value_from(val);
      }
      case 3: {  // text string
        uint64_t n = ReadUInt(ai);
        if (!has(n)) return nullptr;
        std::string s(reinterpret_cast<const char*>(buf_.data() + pos_), n);
        pos_ += n;
        return boost::json::value_from(s);
      }
      case 4: {  // array
        uint64_t n = ReadUInt(ai);
        boost::json::array arr;
        arr.reserve(n);
        for (uint64_t i = 0; i < n && has(1); ++i) {
          arr.push_back(ParseValue());
        }
        return arr;
      }
      case 5: {  // map
        uint64_t n = ReadUInt(ai);
        boost::json::object obj;
        for (uint64_t i = 0; i < n && has(1); ++i) {
          auto key = ParseValue();
          auto val = ParseValue();
          if (key.is_string()) {
            obj[key.as_string()] = val;
          }
        }
        return obj;
      }
      case 7: {  // float / simple
        if (ai == 22) return nullptr;     // null
        if (ai == 25 && has(2)) {         // half float not used
          pos_ += 2;
          return nullptr;
        }
        if (ai == 26 && has(4)) {         // float not used
          pos_ += 4;
          return nullptr;
        }
        if (ai == 27 && has(8)) {         // double
          uint64_t bits = ReadBytes(8);
          double d;
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
  uint8_t type;
  std::vector<uint8_t> payload;  // compressed payload
  boost::json::value json;
};

Frame ReadFrame(websocket::stream<tcp::socket>& ws) {
  beast::flat_buffer buffer;
  ws.read(buffer);
  std::vector<uint8_t> data(buffer.size());
  boost::asio::buffer_copy(boost::asio::buffer(data), buffer.data());
  Frame f{};
  ASSERT_GE(data.size(), 5u);
  f.type = data[0];
  uint32_t len = (static_cast<uint32_t>(data[1]) << 24) |
                 (static_cast<uint32_t>(data[2]) << 16) |
                 (static_cast<uint32_t>(data[3]) << 8) | data[4];
  ASSERT_EQ(len + 5, data.size());
  f.payload.assign(data.begin() + 5, data.end());
  auto inflated = Inflate(f.payload);
  CborReader reader(inflated);
  f.json = reader.Parse();
  return f;
}

}  // namespace

TEST(WsServerTest, SendsNetworkStateAndTickUpdate) {
  senna::config::RuntimeConfig cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
  cfg.network.width = 4;
  cfg.network.height = 4;
  cfg.network.depth = 3;
  cfg.network.density = 0.5;
  cfg.network.neighbor_radius = 1.0F;
  // Use default network; heavy but acceptable for integration-style test.
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);

  constexpr int kPort = 19090;
  senna::interfaces::WsServer server(kPort, &net);
  server.Start();
  std::this_thread::sleep_for(200ms);  // give server time to start

  boost::asio::io_context ioc;
  tcp::resolver resolver(ioc);
  auto results = resolver.resolve("127.0.0.1", std::to_string(kPort));
  websocket::stream<tcp::socket> ws(ioc);
  boost::asio::connect(beast::get_lowest_layer(ws), results);
  ws.handshake("127.0.0.1", "/");

  Frame netstate = ReadFrame(ws);
  ASSERT_EQ(netstate.type, 1);
  ASSERT_TRUE(netstate.json.is_object());
  auto& ns = netstate.json.as_object();
  EXPECT_EQ(ns.at("type").as_string(), "NetworkState");
  const auto expected_neurons =
      static_cast<int64_t>(cfg.network.width * cfg.network.height * cfg.network.depth);
  EXPECT_EQ(ns.at("neuron_count").as_int64(), expected_neurons);

  // Send a command and expect Ack (type 0x10).
  ws.write(boost::asio::buffer(R"({"type":"SetUpdateInterval","ticks":1})"));
  bool got_ack = false;
  for (int i = 0; i < 10 && !got_ack; ++i) {
    Frame f = ReadFrame(ws);
    if (f.type == 0x10) {
      ASSERT_TRUE(f.json.is_object());
      auto& ack = f.json.as_object();
      EXPECT_EQ(ack.at("cmd").as_string(), "SetUpdateInterval");
      EXPECT_EQ(ack.at("ok").as_bool(), true);
      got_ack = true;
    }
  }
  EXPECT_TRUE(got_ack);

  ws.close(websocket::close_code::normal);
  server.Stop();
}

TEST(WsServerTest, StreamsSynapseChunksAndHonorsPauseResume) {
  senna::config::RuntimeConfig cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
  cfg.network.width = 3;
  cfg.network.height = 3;
  cfg.network.depth = 2;
  cfg.network.density = 0.6;
  senna::observability::MetricsCollector metrics;
  senna::network::Network net(cfg.network, &metrics);

  constexpr int kPort = 19091;
  senna::interfaces::WsServer server(kPort, &net);
  server.Start();
  std::this_thread::sleep_for(150ms);

  boost::asio::io_context ioc;
  tcp::resolver resolver(ioc);
  auto results = resolver.resolve("127.0.0.1", std::to_string(kPort));
  websocket::stream<tcp::socket> ws(ioc);
  boost::asio::connect(beast::get_lowest_layer(ws), results);
  ws.handshake("127.0.0.1", "/");

  // Consume NetworkState.
  Frame netstate = ReadFrame(ws);
  ASSERT_EQ(netstate.type, 1);
  auto& ns = netstate.json.as_object();
  const auto expected_neurons =
      static_cast<int64_t>(cfg.network.width * cfg.network.height * cfg.network.depth);
  EXPECT_EQ(ns.at("neuron_count").as_int64(), expected_neurons);

  // Expect at least one synapse chunk.
  Frame chunk = ReadFrame(ws);
  ASSERT_EQ(chunk.type, 2);
  ASSERT_TRUE(chunk.json.is_object());
  auto& ch = chunk.json.as_object();
  EXPECT_EQ(ch.at("type").as_string(), "SynapseChunk");
  EXPECT_GE(ch.at("total").as_int64(), ch.at("synapses").as_array().size());

  // Set update interval to 1 and ensure we get a TickUpdate when running ticks.
  ws.write(boost::asio::buffer(R"({"type":"SetUpdateInterval","ticks":1})"));
  // Run ticks in background to generate snapshots.
  std::atomic<bool> tick_run{true};
  std::thread tick_thread([&]() {
    for (int i = 0; i < 50 && tick_run.load(); ++i) {
      auto syn = net.synapses_ptr();
      net.time_manager().Tick(net.queue(), net.pool(), *syn);
      std::this_thread::sleep_for(5ms);
    }
  });

  bool got_tick = false;
  bool got_ack = false;
  for (int i = 0; i < 15 && (!got_tick || !got_ack); ++i) {
    Frame f = ReadFrame(ws);
    if (f.type == 0x10) {
      got_ack = true;
    } else if (f.type == 3) {
      got_tick = true;
    }
  }
  EXPECT_TRUE(got_ack);
  EXPECT_TRUE(got_tick);

  // Pause streaming; consume for a short window and ensure no TickUpdate arrives.
  ws.write(boost::asio::buffer(R"({"type":"Pause"})"));
  // Wait for pause Ack to ensure command applied.
  for (int i = 0; i < 5; ++i) {
    Frame f = ReadFrame(ws);
    if (f.type == 0x10 && f.json.as_object().at("cmd").as_string() == "Pause") {
      break;
    }
  }
  auto start = std::chrono::steady_clock::now();
  auto last_tick = start;
  int tick_count = 0;
  while (std::chrono::steady_clock::now() - start < 200ms) {
    beast::get_lowest_layer(ws).expires_after(std::chrono::milliseconds(20));
    beast::flat_buffer buf;
    beast::error_code ec;
    ws.read(buf, ec);
    if (!ec) {
      std::vector<uint8_t> msg(buf.size());
      boost::asio::buffer_copy(boost::asio::buffer(msg), buf.data());
      if (!msg.empty() && msg[0] == 3) {
        ++tick_count;
        last_tick = std::chrono::steady_clock::now();
      }
    } else if (ec == websocket::error::timeout ||
               ec == boost::asio::error::operation_aborted) {
      continue;
    } else {
      break;
    }
    if (std::chrono::steady_clock::now() - last_tick > 80ms) {
      break;  // no tick for a while, pause likely active
    }
  }
  EXPECT_LE(tick_count, 1);  // allow at most one in-flight frame

  beast::get_lowest_layer(ws).expires_never();

  // Resume and expect tick updates again.
  ws.write(boost::asio::buffer(R"({"type":"Resume"})"));
  bool got_after_resume = false;
  for (int i = 0; i < 10 && !got_after_resume; ++i) {
    Frame f = ReadFrame(ws);
    if (f.type == 3) {
      got_after_resume = true;
    }
  }
  EXPECT_TRUE(got_after_resume);

  tick_run.store(false);
  if (tick_thread.joinable()) tick_thread.join();
  ws.close(websocket::close_code::normal);
  server.Stop();
}

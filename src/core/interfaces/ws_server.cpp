#include "core/interfaces/ws_server.hpp"

#include <zlib.h>

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>
#include <chrono>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/neural/neuron_pool.hpp"
#include "core/spatial/lattice.hpp"
#include "core/synaptic/synapse_index.hpp"
#include "core/temporal/time_manager.hpp"

namespace senna::interfaces {

namespace beast = boost::beast;
namespace websocket = beast::websocket;
using tcp = boost::asio::ip::tcp;

WsServer::WsServer(int port, network::Network* net) : port_(port), net_(net) {}

WsServer::~WsServer() { Stop(); }

void WsServer::Start() {
  stop_.store(false);
  snapshot_thread_ = std::thread([this]() {
    while (!stop_.load()) {
      SnapshotNetwork();
      std::this_thread::sleep_for(snapshot_period_);
    }
  });
  server_thread_ = std::thread([this]() { Run(); });
}

void WsServer::SnapshotNetwork() {
  // Build NetworkState JSON
  boost::json::object obj;
  const auto& lattice = net_->lattice();
  const auto& pool = net_->pool();
  auto syn = net_->synapses_ptr();
  std::unordered_set<int32_t> output_set(net_->output_ids().begin(),
                                         net_->output_ids().end());

  obj["type"] = "NetworkState";
  obj["neuron_count"] = pool.size();
  obj["synapse_count"] = syn->synapse_count();

  boost::json::array neurons;
  neurons.reserve(pool.size());
  for (int id = 0; id < pool.size(); ++id) {
    auto [x, y, z] = lattice.CoordsOf(id);
    const char* t = nullptr;
    if (output_set.contains(id)) {
      t = "O";
    } else if (pool.type(id) == neural::NeuronType::Excitatory) {
      t = "E";
    } else {
      t = "I";
    }
    boost::json::object n;
    n["id"] = id;
    n["x"] = x;
    n["y"] = y;
    n["z"] = z;
    n["type"] = t;
    neurons.push_back(std::move(n));
  }
  obj["neurons"] = std::move(neurons);

  boost::json::array outs;
  for (int id : net_->output_ids()) {
    outs.push_back(id);
  }
  obj["output_ids"] = std::move(outs);

  auto buf = std::make_shared<std::vector<uint8_t>>();
  *buf = PackCborDeflate(1, obj);
  netstate_buf_.store(buf);

  // Build synapse chunks
  constexpr size_t kChunk = 5000;
  const auto& vec = syn->synapses();
  const size_t max_send = std::min<size_t>(vec.size(), 20000);
  auto chunk_vec = std::make_shared<std::vector<std::vector<uint8_t>>>();
  std::unordered_set<int32_t> out_set(net_->output_ids().begin(),
                                      net_->output_ids().end());
  boost::json::object chunk;
  chunk["type"] = "SynapseChunk";
  chunk["total"] = max_send;
  for (size_t start = 0; start < max_send; start += kChunk) {
    size_t end = std::min(max_send, start + kChunk);
    boost::json::array arr;
    arr.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      boost::json::object s;
      s["pre"] = vec[i].pre_id;
      s["post"] = vec[i].post_id;
      bool wta =
          out_set.contains(vec[i].pre_id) && out_set.contains(vec[i].post_id);
      s["wta"] = wta;
      arr.push_back(std::move(s));
    }
    chunk["start"] = static_cast<uint64_t>(start);
    chunk["synapses"] = std::move(arr);
    chunk["last"] = (end == max_send);
    chunk_vec->push_back(PackCborDeflate(2, chunk));
  }
  syn_chunks_buf_.store(chunk_vec);
}

void WsServer::Stop() {
  if (stop_.exchange(true)) {
    return;  // already stopped
  }
  // Unblock the synchronous accept() by connecting to ourselves.
  // acceptor_->close() from another thread is not safe for sync accept.
  try {
    boost::asio::io_context tmp_ioc;
    tcp::socket wake_sock(tmp_ioc);
    wake_sock.connect(
        tcp::endpoint(boost::asio::ip::address_v4::loopback(), port_));
  } catch (const std::exception& e) {
    (void)e;
  }
  if (ioc_) {
    ioc_->stop();
  }
  if (snapshot_thread_.joinable()) {
    snapshot_thread_.join();
  }
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
}

void WsServer::Run() {
  try {
    ioc_ = std::make_unique<boost::asio::io_context>();
    acceptor_ =
        std::make_unique<tcp::acceptor>(*ioc_, tcp::endpoint(tcp::v4(), port_));

    while (!stop_.load()) {
      tcp::socket socket(*ioc_);
      boost::system::error_code ec;
      acceptor_->accept(socket, ec);  // NOLINT(bugprone-unused-return-value)
      if (ec) {
        if (stop_.load()) {
          break;
        }
        continue;
      }
      auto guard = std::make_shared<int>(0);  // keeps session alive
      std::thread(&WsServer::HandleSession, this, guard, std::move(socket))
          .detach();
    }
  } catch (const std::exception& e) {
    // Log and keep core alive
    (void)e;
  }
}

void WsServer::HandleSession(const std::shared_ptr<void>& /*guard*/,
                             tcp::socket socket) {
  try {
    auto ws =
        std::make_shared<websocket::stream<tcp::socket>>(std::move(socket));
    ws->set_option(
        websocket::stream_base::timeout::suggested(beast::role_type::server));
    ws->accept();

    // Send static network snapshot once.
    ws->binary(true);
    if (auto buf = netstate_buf_.load(); buf && !buf->empty()) {
      ws->write(boost::asio::buffer(*buf));
    } else {
      auto snapshot = BuildNetworkState();
      auto payload = PackCborDeflate(1, boost::json::parse(snapshot));
      ws->write(boost::asio::buffer(payload));
    }
    SendSynapseChunks(*ws);

    // Reader for control commands.
    std::thread reader([this, ws]() {
      beast::flat_buffer buf;
      while (!stop_.load()) {
        boost::system::error_code ec;
        ws->read(buf, ec);
        if (ec) {
          break;
        }
        auto data = beast::buffers_to_string(buf.data());
        HandleCommand(data, *ws);
        buf.consume(buf.size());
      }
    });

    // Periodic lightweight tick update (time + counts).
    while (!stop_.load()) {
      if (!paused_.load()) {
        auto snap = net_->time_manager().SnapshotPtr();
        auto last_tick = last_sent_tick_.load();
        if (snap && snap->tick > last_tick &&
            (snap->tick - last_tick) >=
                static_cast<uint64_t>(update_interval_ticks_.load())) {
          last_sent_tick_.store(snap->tick);
          auto update = BuildTickUpdate();
          auto packed = PackCborDeflate(3, boost::json::parse(update));
          ws->write(boost::asio::buffer(packed));
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (reader.joinable()) {
      reader.join();
    }
  } catch (const std::exception& e) {
    // client disconnected or other error; ignore but keep thread alive
    (void)e;
  }
}

std::string WsServer::BuildNetworkState() const {
  boost::json::object obj;
  const auto& lattice = net_->lattice();
  const auto& pool = net_->pool();
  auto syn = net_->synapses_ptr();
  std::unordered_set<int32_t> output_set(net_->output_ids().begin(),
                                         net_->output_ids().end());

  obj["type"] = "NetworkState";
  obj["neuron_count"] = pool.size();
  obj["synapse_count"] = syn->synapse_count();

  boost::json::array neurons;
  neurons.reserve(pool.size());
  for (int id = 0; id < pool.size(); ++id) {
    auto [x, y, z] = lattice.CoordsOf(id);
    const char* t = nullptr;
    if (output_set.contains(id)) {
      t = "O";
    } else if (pool.type(id) == neural::NeuronType::Excitatory) {
      t = "E";
    } else {
      t = "I";
    }
    boost::json::object n;
    n["id"] = id;
    n["x"] = x;
    n["y"] = y;
    n["z"] = z;
    n["type"] = t;
    neurons.push_back(std::move(n));
  }
  obj["neurons"] = std::move(neurons);

  boost::json::array outs;
  for (int id : net_->output_ids()) {
    outs.push_back(id);
  }
  obj["output_ids"] = std::move(outs);

  return boost::json::serialize(obj);
}

std::string WsServer::BuildTickUpdate() const {
  auto snap = net_->time_manager().SnapshotPtr();
  if (!snap) {
    boost::json::object empty;
    empty["type"] = "TickUpdate";
    empty["t_ms"] = 0;
    empty["phase"] = net_->phase();
    empty["sleep_pressure"] = net_->sleep_pressure();
    return boost::json::serialize(empty);
  }
  auto fired = snap->fired;
  auto spikes = snap->spikes;
  boost::json::object obj;
  obj["type"] = "TickUpdate";
  obj["t_ms"] = snap->t_ms;
  obj["phase"] = net_->phase();
  obj["sleep_pressure"] = net_->sleep_pressure();
  obj["tick"] = snap->tick;
  obj["label"] = current_label_.load();

  boost::json::array fired_arr;
  fired_arr.reserve(fired.size());
  for (int id : fired) {
    boost::json::object f;
    f["id"] = id;
    f["V"] = net_->pool().V(id);
    fired_arr.push_back(std::move(f));
  }
  obj["fired"] = std::move(fired_arr);

  boost::json::array spikes_arr;
  spikes_arr.reserve(spikes.size());
  for (const auto& s : spikes) {
    boost::json::object sp;
    sp["id"] = s.first;
    sp["t"] = s.second;
    spikes_arr.push_back(std::move(sp));
  }
  obj["spikes"] = std::move(spikes_arr);

  return boost::json::serialize(obj);
}

void WsServer::HandleCommand(
    std::string_view msg,
    boost::beast::websocket::stream<boost::asio::ip::tcp::socket>& ws) {
  try {
    auto j = boost::json::parse(msg);
    if (!j.is_object()) {
      return;
    }
    auto& o = j.as_object();
    if (!o.contains("type")) {
      return;
    }
    auto type = o["type"].as_string();
    boost::json::object ack;
    ack["type"] = "Ack";
    ack["cmd"] = type;
    ack["ok"] = false;
    if (type == "SetUpdateInterval") {
      if (auto* ticks = o.if_contains("ticks");
          ticks != nullptr && ticks->is_int64()) {
        int val = static_cast<int>(ticks->as_int64());
        if (val > 0 && val < 10000) {
          update_interval_ticks_.store(val);
        }
        ack["ticks"] = val;
        ack["ok"] = true;
      }
    } else if (type == "Pause") {
      paused_.store(true);
      ack["ok"] = true;
    } else if (type == "Resume") {
      paused_.store(false);
      ack["ok"] = true;
    } else if (type == "SetLabel") {
      if (auto* label = o.if_contains("label");
          label != nullptr && label->is_int64()) {
        int val = static_cast<int>(label->as_int64());
        if (val >= 0 && val <= 9) {
          current_label_.store(val);
        }
        ack["label"] = val;
        ack["ok"] = true;
      }
    }
    auto packed = PackCborDeflate(0x10, ack);
    ws.write(boost::asio::buffer(packed));
  } catch (const std::exception& e) {
    (void)e;
  }
}

void WsServer::SendSynapseChunks(websocket::stream<tcp::socket>& ws) {
  if (auto chunks = syn_chunks_buf_.load(); chunks && !chunks->empty()) {
    for (const auto& c : *chunks) {
      ws.write(boost::asio::buffer(c));
    }
    return;
  }
  constexpr size_t kChunk = 5000;
  auto syn = net_->synapses_ptr();
  const auto& vec = syn->synapses();
  const size_t max_send = std::min<size_t>(vec.size(), 20000);  // LOD cap
  boost::json::object chunk;
  chunk["type"] = "SynapseChunk";
  chunk["total"] = max_send;
  for (size_t start = 0; start < max_send; start += kChunk) {
    size_t end = std::min(max_send, start + kChunk);
    boost::json::array arr;
    arr.reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      boost::json::object s;
      s["pre"] = vec[i].pre_id;
      s["post"] = vec[i].post_id;
      arr.push_back(std::move(s));
    }
    chunk["start"] = static_cast<uint64_t>(start);
    chunk["synapses"] = std::move(arr);
    chunk["last"] = (end == vec.size());
    auto packed = PackCborDeflate(2, chunk);
    ws.write(boost::asio::buffer(packed));
  }
}

void WsServer::EncodeCbor(const boost::json::value& v,
                          std::vector<uint8_t>& out) const {
  using boost::json::array;
  using boost::json::object;
  switch (v.kind()) {
    case boost::json::kind::int64: {
      int64_t n = v.get_int64();
      if (n >= 0) {
        if (n < 24) {
          out.push_back(static_cast<uint8_t>(n));
        } else if (n <= 0xFF) {
          out.push_back(0x18);
          out.push_back(static_cast<uint8_t>(n));
        } else if (n <= 0xFFFF) {
          out.push_back(0x19);
          out.push_back(static_cast<uint8_t>(n >> 8));
          out.push_back(static_cast<uint8_t>(n));
        } else {
          out.push_back(0x1b);
          for (int i = 7; i >= 0; --i) {
            out.push_back(static_cast<uint8_t>(
                (static_cast<uint64_t>(n) >> (8 * i)) & 0xFF));
          }
        }
      } else {
        int64_t m = -1 - n;
        if (m < 24) {
          out.push_back(static_cast<uint8_t>(0x20 | m));
        } else if (m <= 0xFF) {
          out.push_back(0x38);
          out.push_back(static_cast<uint8_t>(m));
        } else if (m <= 0xFFFF) {
          out.push_back(0x39);
          out.push_back(static_cast<uint8_t>(m >> 8));
          out.push_back(static_cast<uint8_t>(m));
        } else {
          out.push_back(0x3b);
          for (int i = 7; i >= 0; --i) {
            out.push_back(static_cast<uint8_t>(
                (static_cast<uint64_t>(m) >> (8 * i)) & 0xFF));
          }
        }
      }
      break;
    }
    case boost::json::kind::uint64: {
      uint64_t n = v.get_uint64();
      if (n < 24) {
        out.push_back(static_cast<uint8_t>(n));
      } else if (n <= 0xFF) {
        out.push_back(0x18);
        out.push_back(static_cast<uint8_t>(n));
      } else if (n <= 0xFFFF) {
        out.push_back(0x19);
        out.push_back(static_cast<uint8_t>(n >> 8));
        out.push_back(static_cast<uint8_t>(n));
      } else {
        out.push_back(0x1b);
        for (int i = 7; i >= 0; --i) {
          out.push_back(static_cast<uint8_t>((n >> (8 * i)) & 0xFF));
        }
      }
      break;
    }
    case boost::json::kind::double_: {
      out.push_back(0xfb);
      uint64_t bits = 0;
      double d = v.get_double();
      std::memcpy(&bits, &d, sizeof(bits));
      for (int i = 7; i >= 0; --i) {
        out.push_back(static_cast<uint8_t>((bits >> (8 * i)) & 0xFF));
      }
      break;
    }
    case boost::json::kind::string: {
      const auto& s = v.get_string();
      uint64_t n = s.size();
      if (n < 24) {
        out.push_back(static_cast<uint8_t>(0x60 | n));
      } else {
        out.push_back(0x78);
        out.push_back(static_cast<uint8_t>(n));
      }
      out.insert(out.end(), s.begin(), s.end());
      break;
    }
    case boost::json::kind::array: {
      const auto& a = v.get_array();
      uint64_t n = a.size();
      if (n < 24) {
        out.push_back(static_cast<uint8_t>(0x80 | n));
      } else {
        out.push_back(0x98);
        out.push_back(static_cast<uint8_t>(n));
      }
      for (const auto& e : a) {
        EncodeCbor(e, out);
      }
      break;
    }
    case boost::json::kind::object: {
      const auto& o = v.get_object();
      uint64_t n = o.size();
      if (n < 24) {
        out.push_back(static_cast<uint8_t>(0xa0 | n));
      } else {
        out.push_back(0xb8);
        out.push_back(static_cast<uint8_t>(n));
      }
      for (const auto& kv : o) {
        boost::json::string k = kv.key();
        boost::json::value key_val(k);
        EncodeCbor(key_val, out);
        EncodeCbor(kv.value(), out);
      }
      break;
    }
    case boost::json::kind::bool_:
      out.push_back(v.get_bool() ? 0xf5 : 0xf4);
      break;
    case boost::json::kind::null:
    default:
      out.push_back(0xf6);  // null
      break;
  }
}

std::vector<uint8_t> WsServer::PackCborDeflate(
    uint8_t type, const boost::json::value& doc) const {
  std::vector<uint8_t> out;
  std::vector<uint8_t> cbor;
  EncodeCbor(doc, cbor);
  auto src_len = static_cast<uLong>(cbor.size());
  auto bound = compressBound(src_len);
  out.resize(1 + 4 + bound);
  out[0] = type;
  // placeholder for size
  auto* dest = reinterpret_cast<Bytef*>(out.data() + 5);  // NOLINT
  auto dest_len = bound;
  if (compress2(dest, &dest_len,
                reinterpret_cast<const Bytef*>(cbor.data()),  // NOLINT
                src_len, Z_BEST_SPEED) != Z_OK) {
    out.resize(1 + 4 + src_len);
    std::memcpy(out.data() + 5, cbor.data(), src_len);
    dest_len = src_len;
  }
  auto len = static_cast<uint32_t>(dest_len);
  out[1] = static_cast<uint8_t>((len >> 24) & 0xFF);
  out[2] = static_cast<uint8_t>((len >> 16) & 0xFF);
  out[3] = static_cast<uint8_t>((len >> 8) & 0xFF);
  out[4] = static_cast<uint8_t>(len & 0xFF);
  out.resize(5 + dest_len);
  return out;
}

}  // namespace senna::interfaces

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "core/decoding/first_spike_decoder.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"

namespace grpc {
class Server;
}

namespace senna::interfaces {

// Pending stimulus that has been injected but not yet decoded.
struct StimulusRecord {
  uint64_t id{0};
  int32_t label{-1};
  float inject_time_ms{0};
  float duration_ms{50};
  std::optional<int> predicted_class;
  double confidence{0};
  uint32_t latency_ms{0};
  bool ready{false};
};

// gRPC server for the SennaCore service (Step 14.2).
// Runs in its own thread pool. Accepts stimuli, places them into the
// Network's event queue, and returns predictions via the decoder.
class GrpcServer {
 public:
  GrpcServer(int port, network::Network* net,
             observability::MetricsCollector* metrics = nullptr);
  ~GrpcServer();

  void Start();
  void Stop();

  // Access stimulus records (thread-safe). Used by tests and internally.
  StimulusRecord* FindStimulus(uint64_t id);
  uint64_t InjectStimulus(const std::vector<uint32_t>& pixels, int32_t label,
                          uint32_t duration_ms);
  void TryDecode(uint64_t stimulus_id);

  // Notify waiting StreamPredictions clients.
  void NotifyPrediction(uint64_t stimulus_id, int predicted_class,
                        double confidence, uint32_t latency_ms);

  bool shutting_down() const { return stop_.load(); }
  network::Network* net() const { return net_; }
  observability::MetricsCollector* metrics() const { return metrics_; }

 private:
  friend class SennaCoreServiceImpl;
  int port_;
  network::Network* net_;
  observability::MetricsCollector* metrics_;
  std::atomic<bool> stop_{false};
  std::unique_ptr<grpc::Server> server_;
  std::thread server_thread_;

  // Stimulus tracking.
  std::mutex stimuli_mutex_;
  std::vector<StimulusRecord> stimuli_;
  uint64_t next_stimulus_id_{1};

  // Prediction stream notification.
  std::mutex stream_mutex_;
  std::condition_variable stream_cv_;
  struct PredictionNotification {
    uint64_t stimulus_id;
    int predicted_class;
    double confidence;
    uint32_t latency_ms;
  };
  std::deque<PredictionNotification> pending_predictions_;
};

}  // namespace senna::interfaces

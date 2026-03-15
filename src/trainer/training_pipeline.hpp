#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "trainer/config.hpp"
#include "trainer/mnist_loader.hpp"

// Forward declarations to avoid pulling in gRPC/proto headers.
namespace grpc {
class Channel;
}

namespace senna::trainer {

// Result of a single training/test epoch.
struct EpochResult {
  int epoch{0};
  int correct{0};
  int total{0};
  double accuracy{0.0};
  bool is_test{false};
};

// Callback for epoch completion.
using EpochCallback = std::function<void(const EpochResult&)>;

// Training pipeline that connects to senna-core via gRPC and drives
// MNIST training/testing epochs.
class TrainingPipeline {
 public:
  explicit TrainingPipeline(TrainerConfig cfg);
  ~TrainingPipeline();

  // Connect to senna-core. Returns false if connection fails.
  bool Connect();

  // Run training epochs on the given datasets.
  void Train(const MnistLoader& train_data, const MnistLoader& test_data);

  // Run test-only evaluation.
  EpochResult Test(const MnistLoader& test_data);

  // Pause/resume/stop training (does not affect the core).
  void Pause();
  void Resume();
  void Stop();

  bool is_running() const { return running_.load(); }
  bool is_paused() const { return paused_.load(); }

  void set_epoch_callback(EpochCallback cb) { epoch_cb_ = std::move(cb); }

 private:
  // Run one epoch over dataset. If supervise=true, sends correction signal.
  EpochResult RunEpoch(const MnistLoader& data, int epoch_num, bool supervise);

  // Inject stimulus via gRPC, returns stimulus_id (0 on failure).
  uint64_t InjectStimulus(const MnistSample& sample);

  // Poll for prediction result, returns predicted class (-1 on timeout).
  int WaitForPrediction(uint64_t stimulus_id);

  // Send supervision signal.
  void Supervise(uint64_t stimulus_id, int correct_class);

  // Report accuracy to core.
  void ReportAccuracy(double train_acc, double test_acc);

  // Spin while paused (respects stop).
  void WaitWhilePaused();

  TrainerConfig cfg_;
  std::shared_ptr<grpc::Channel> channel_;
  // Use void* to avoid including generated header in this header.
  // The .cpp casts to SennaCore::Stub*.
  void* stub_{nullptr};
  std::atomic<bool> running_{false};
  std::atomic<bool> paused_{false};
  std::atomic<bool> stop_requested_{false};
  EpochCallback epoch_cb_;
};

}  // namespace senna::trainer

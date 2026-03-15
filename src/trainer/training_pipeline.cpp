#include "trainer/training_pipeline.hpp"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <iostream>
#include <thread>

#include "senna.grpc.pb.h"
#include "senna.pb.h"

namespace senna::trainer {

TrainingPipeline::TrainingPipeline(TrainerConfig cfg) : cfg_(std::move(cfg)) {}

TrainingPipeline::~TrainingPipeline() {
  Stop();
  delete static_cast<senna::SennaCore::Stub*>(stub_);
}

bool TrainingPipeline::Connect() {
  std::string addr = cfg_.core_host + ":" + std::to_string(cfg_.core_port);
  channel_ = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());

  // Wait for connection with timeout.
  auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(10);
  if (!channel_->WaitForConnected(deadline)) {
    std::cerr << "TrainingPipeline: failed to connect to " << addr << "\n";
    return false;
  }

  auto* raw_stub = new senna::SennaCore::Stub(channel_);
  stub_ = raw_stub;
  std::cout << "TrainingPipeline: connected to " << addr << "\n";
  return true;
}

void TrainingPipeline::Train(const MnistLoader& train_data,
                             const MnistLoader& test_data) {
  running_.store(true);
  stop_requested_.store(false);

  for (int epoch = 1; epoch <= cfg_.epochs; ++epoch) {
    if (stop_requested_.load()) {
      break;
    }

    std::cout << "=== Epoch " << epoch << "/" << cfg_.epochs
              << " (train) ===\n";
    auto train_result = RunEpoch(train_data, epoch, true);
    train_result.is_test = false;
    if (epoch_cb_) {
      epoch_cb_(train_result);
    }

    if (stop_requested_.load()) {
      break;
    }

    // Test evaluation after each training epoch.
    std::cout << "=== Epoch " << epoch << "/" << cfg_.epochs << " (test) ===\n";
    auto test_result = RunEpoch(test_data, epoch, false);
    test_result.is_test = true;
    if (epoch_cb_) {
      epoch_cb_(test_result);
    }

    ReportAccuracy(train_result.accuracy, test_result.accuracy);

    std::cout << "Epoch " << epoch << " train_acc=" << train_result.accuracy
              << " test_acc=" << test_result.accuracy << "\n";
  }

  running_.store(false);
}

EpochResult TrainingPipeline::Test(const MnistLoader& test_data) {
  running_.store(true);
  stop_requested_.store(false);

  auto result = RunEpoch(test_data, 0, false);
  result.is_test = true;
  if (epoch_cb_) {
    epoch_cb_(result);
  }

  running_.store(false);
  return result;
}

void TrainingPipeline::Pause() { paused_.store(true); }
void TrainingPipeline::Resume() { paused_.store(false); }

void TrainingPipeline::Stop() {
  stop_requested_.store(true);
  paused_.store(false);  // unblock WaitWhilePaused
}

EpochResult TrainingPipeline::RunEpoch(const MnistLoader& data, int epoch_num,
                                       bool supervise) {
  EpochResult result;
  result.epoch = epoch_num;

  int limit = static_cast<int>(data.size());
  if (supervise && cfg_.max_train_samples > 0) {
    limit = std::min(limit, cfg_.max_train_samples);
  }
  if (!supervise && cfg_.max_test_samples > 0) {
    limit = std::min(limit, cfg_.max_test_samples);
  }

  for (int i = 0; i < limit; ++i) {
    if (stop_requested_.load()) {
      break;
    }
    WaitWhilePaused();
    if (stop_requested_.load()) {
      break;
    }

    const auto& sample = data[i];
    uint64_t sid = InjectStimulus(sample);
    if (sid == 0) {
      std::cerr << "  sample " << i << ": inject failed\n";
      continue;
    }

    // Wait for presentation window + a little extra.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(cfg_.presentation_ms));

    int predicted = WaitForPrediction(sid);
    bool correct = (predicted == static_cast<int>(sample.label));
    if (correct) {
      ++result.correct;
    }
    ++result.total;

    if (supervise && !correct && predicted >= 0) {
      Supervise(sid, static_cast<int>(sample.label));
    }

    // Inter-stimulus interval.
    if (cfg_.inter_stimulus_ms > 0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(cfg_.inter_stimulus_ms));
    }

    // Progress every 100 samples.
    if ((i + 1) % 100 == 0) {
      double running_acc =
          result.total > 0 ? static_cast<double>(result.correct) / result.total
                           : 0.0;
      std::cout << "  [" << (i + 1) << "/" << limit << "] acc=" << running_acc
                << "\n";
    }
  }

  result.accuracy = result.total > 0
                        ? static_cast<double>(result.correct) / result.total
                        : 0.0;
  return result;
}

uint64_t TrainingPipeline::InjectStimulus(const MnistSample& sample) {
  auto* stub = static_cast<senna::SennaCore::Stub*>(stub_);

  senna::StimulusRequest req;
  for (uint8_t px : sample.pixels) {
    req.add_pixels(static_cast<uint32_t>(px));
  }
  req.set_label(static_cast<int32_t>(sample.label));
  req.set_duration_ms(cfg_.presentation_ms);

  senna::StimulusResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(cfg_.prediction_timeout_ms));

  auto status = stub->InjectStimulus(&ctx, req, &resp);
  if (!status.ok()) {
    std::cerr << "InjectStimulus RPC failed: " << status.error_message()
              << "\n";
    return 0;
  }
  if (!resp.accepted()) {
    // Might be during sleep phase — not an error.
    return 0;
  }
  return resp.stimulus_id();
}

int TrainingPipeline::WaitForPrediction(uint64_t stimulus_id) {
  auto* stub = static_cast<senna::SennaCore::Stub*>(stub_);
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(cfg_.prediction_timeout_ms);

  while (std::chrono::steady_clock::now() < deadline) {
    senna::PredictionRequest req;
    req.set_stimulus_id(stimulus_id);

    senna::PredictionResponse resp;
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() +
                     std::chrono::milliseconds(cfg_.prediction_timeout_ms));

    auto status = stub->GetPrediction(&ctx, req, &resp);
    if (!status.ok()) {
      return -1;
    }
    if (resp.ready()) {
      return resp.predicted_class();
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(cfg_.prediction_poll_ms));
  }

  return -1;  // timeout
}

void TrainingPipeline::Supervise(uint64_t stimulus_id, int correct_class) {
  auto* stub = static_cast<senna::SennaCore::Stub*>(stub_);

  senna::SupervisionRequest req;
  req.set_stimulus_id(stimulus_id);
  req.set_correct_class(correct_class);

  senna::SupervisionResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(cfg_.prediction_timeout_ms));

  auto status = stub->Supervise(&ctx, req, &resp);
  if (!status.ok()) {
    std::cerr << "Supervise RPC failed: " << status.error_message() << "\n";
  }
}

void TrainingPipeline::ReportAccuracy(double train_acc, double test_acc) {
  auto* stub = static_cast<senna::SennaCore::Stub*>(stub_);

  senna::AccuracyRequest req;
  req.set_train_accuracy(train_acc);
  req.set_test_accuracy(test_acc);

  google::protobuf::Empty resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

  stub->ReportAccuracy(&ctx, req, &resp);
}

void TrainingPipeline::WaitWhilePaused() {
  while (paused_.load() && !stop_requested_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

}  // namespace senna::trainer

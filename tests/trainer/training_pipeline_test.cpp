#include "trainer/training_pipeline.hpp"

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <chrono>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include "core/config/runtime_config.hpp"
#include "core/interfaces/grpc_server.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"
#include "senna.grpc.pb.h"
#include "senna.pb.h"
#include "trainer/config.hpp"
#include "trainer/mnist_loader.hpp"

using namespace std::chrono_literals;

namespace {

// Helper: build a small network + gRPC server for integration tests.
struct IntegrationEnv {
  senna::config::RuntimeConfig core_cfg;
  senna::observability::MetricsCollector metrics;
  std::unique_ptr<senna::network::Network> net;
  std::unique_ptr<senna::interfaces::GrpcServer> grpc_server;
  int port;

  explicit IntegrationEnv(int p) : port(p) {
    core_cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
    core_cfg.network.width = 4;
    core_cfg.network.height = 4;
    core_cfg.network.depth = 3;
    core_cfg.network.density = 1.0;
    core_cfg.network.neighbor_radius = 1.0F;
    net = std::make_unique<senna::network::Network>(core_cfg.network, &metrics);
    grpc_server = std::make_unique<senna::interfaces::GrpcServer>(
        port, net.get(), &metrics);
    grpc_server->Start();
    std::this_thread::sleep_for(300ms);
  }

  ~IntegrationEnv() { grpc_server->Stop(); }

  // Advance virtual time by running ticks.
  void RunTicks(int n) {
    for (int i = 0; i < n; ++i) {
      auto syn = net->synapses_ptr();
      net->time_manager().Tick(net->queue(), net->pool(), *syn);
    }
  }
};

// Create synthetic MNIST test data in memory.
senna::trainer::MnistLoader MakeSyntheticData(int count, uint8_t fill = 128) {
  // Write temporary IDX files and load them.
  std::string img_path = "/tmp/pipeline_test_images.idx3";
  std::string lbl_path = "/tmp/pipeline_test_labels.idx1";

  {
    std::ofstream out(img_path, std::ios::binary);
    auto writeU32 = [&](uint32_t v) {
      uint8_t b[4] = {static_cast<uint8_t>((v >> 24) & 0xFF),
                      static_cast<uint8_t>((v >> 16) & 0xFF),
                      static_cast<uint8_t>((v >> 8) & 0xFF),
                      static_cast<uint8_t>(v & 0xFF)};
      out.write(reinterpret_cast<const char*>(b), 4);
    };
    writeU32(0x00000803);
    writeU32(count);
    writeU32(28);
    writeU32(28);
    std::vector<uint8_t> pixels(784, fill);
    for (int i = 0; i < count; ++i) {
      pixels[0] = static_cast<uint8_t>(i % 256);
      out.write(reinterpret_cast<const char*>(pixels.data()), 784);
    }
  }
  {
    std::ofstream out(lbl_path, std::ios::binary);
    auto writeU32 = [&](uint32_t v) {
      uint8_t b[4] = {static_cast<uint8_t>((v >> 24) & 0xFF),
                      static_cast<uint8_t>((v >> 16) & 0xFF),
                      static_cast<uint8_t>((v >> 8) & 0xFF),
                      static_cast<uint8_t>(v & 0xFF)};
      out.write(reinterpret_cast<const char*>(b), 4);
    };
    writeU32(0x00000801);
    writeU32(count);
    for (int i = 0; i < count; ++i) {
      uint8_t label = static_cast<uint8_t>(i % 10);
      out.write(reinterpret_cast<const char*>(&label), 1);
    }
  }

  senna::trainer::MnistLoader loader;
  loader.Load(img_path, lbl_path);
  return loader;
}

}  // namespace

// 15.5.8 Pipeline connects to core and injects a stimulus.
TEST(TrainingPipelineTest, ConnectsAndInjectsStimulus) {
  IntegrationEnv env(19200);

  senna::trainer::TrainerConfig cfg;
  cfg.core_host = "127.0.0.1";
  cfg.core_port = 19200;
  cfg.presentation_ms = 10;
  cfg.prediction_timeout_ms = 200;
  cfg.prediction_poll_ms = 5;

  senna::trainer::TrainingPipeline pipeline(cfg);
  ASSERT_TRUE(pipeline.Connect());

  // Inject a single sample directly via the gRPC stub.
  auto data = MakeSyntheticData(1);
  auto result = pipeline.Test(data);
  EXPECT_EQ(result.total, 1);
  EXPECT_TRUE(result.is_test);
}

// 15.5.9 Pipeline runs a short training epoch with supervision.
TEST(TrainingPipelineTest, TrainEpochWithSupervision) {
  IntegrationEnv env(19201);

  // Run some ticks in background to advance virtual time.
  std::thread ticker([&] {
    for (int i = 0; i < 2000 && !env.grpc_server->shutting_down(); ++i) {
      env.RunTicks(1);
      std::this_thread::sleep_for(1ms);
    }
  });

  senna::trainer::TrainerConfig cfg;
  cfg.core_host = "127.0.0.1";
  cfg.core_port = 19201;
  cfg.epochs = 1;
  cfg.presentation_ms = 5;
  cfg.inter_stimulus_ms = 2;
  cfg.prediction_timeout_ms = 100;
  cfg.prediction_poll_ms = 2;
  cfg.max_train_samples = 3;
  cfg.max_test_samples = 2;

  senna::trainer::TrainingPipeline pipeline(cfg);
  ASSERT_TRUE(pipeline.Connect());

  auto train_data = MakeSyntheticData(3);
  auto test_data = MakeSyntheticData(2);

  int epoch_count = 0;
  pipeline.set_epoch_callback(
      [&](const senna::trainer::EpochResult&) { ++epoch_count; });

  pipeline.Train(train_data, test_data);

  // Should have 2 callbacks: 1 train epoch + 1 test epoch.
  EXPECT_EQ(epoch_count, 2);

  env.grpc_server->Stop();
  ticker.join();
}

// 15.5.10 Stop halts pipeline mid-epoch.
TEST(TrainingPipelineTest, StopHaltsPipeline) {
  IntegrationEnv env(19202);

  std::thread ticker([&] {
    for (int i = 0; i < 5000 && !env.grpc_server->shutting_down(); ++i) {
      env.RunTicks(1);
      std::this_thread::sleep_for(1ms);
    }
  });

  senna::trainer::TrainerConfig cfg;
  cfg.core_host = "127.0.0.1";
  cfg.core_port = 19202;
  cfg.epochs = 100;
  cfg.presentation_ms = 5;
  cfg.inter_stimulus_ms = 2;
  cfg.prediction_timeout_ms = 50;
  cfg.prediction_poll_ms = 2;
  cfg.max_train_samples = 1000;
  cfg.max_test_samples = 100;

  senna::trainer::TrainingPipeline pipeline(cfg);
  ASSERT_TRUE(pipeline.Connect());

  auto train_data = MakeSyntheticData(1000);
  auto test_data = MakeSyntheticData(100);

  // Start training in a thread and stop after a short delay.
  std::thread train_thread([&] { pipeline.Train(train_data, test_data); });

  std::this_thread::sleep_for(200ms);
  pipeline.Stop();
  train_thread.join();

  // Pipeline should have stopped (not completed all 100 epochs).
  EXPECT_FALSE(pipeline.is_running());

  env.grpc_server->Stop();
  ticker.join();
}

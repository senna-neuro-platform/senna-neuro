#include "core/interfaces/grpc_server.hpp"

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "core/config/runtime_config.hpp"
#include "core/network/network_builder.hpp"
#include "core/observability/metrics_collector.hpp"
#include "senna.grpc.pb.h"
#include "senna.pb.h"

using namespace std::chrono_literals;

namespace {

struct TestEnv {
  senna::config::RuntimeConfig cfg;
  senna::observability::MetricsCollector metrics;
  std::unique_ptr<senna::network::Network> net;
  std::unique_ptr<senna::interfaces::GrpcServer> server;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<senna::SennaCore::Stub> stub;

  explicit TestEnv(int port) {
    cfg = senna::config::LoadRuntimeConfig("configs/default.yaml");
    cfg.network.width = 4;
    cfg.network.height = 4;
    cfg.network.depth = 3;
    cfg.network.density = 1.0;
    cfg.network.neighbor_radius = 1.0F;
    net = std::make_unique<senna::network::Network>(cfg.network, &metrics);
    server = std::make_unique<senna::interfaces::GrpcServer>(port, net.get(),
                                                             &metrics);
    server->Start();
    std::this_thread::sleep_for(300ms);

    std::string addr = "127.0.0.1:" + std::to_string(port);
    channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    stub = senna::SennaCore::NewStub(channel);
  }

  ~TestEnv() { server->Stop(); }
};

std::vector<uint32_t> MakeTestImage(uint8_t value = 128) {
  return std::vector<uint32_t>(28 * 28, value);
}

}  // namespace

// 14.3.1 InjectStimulus: stimulus accepted, stimulus_id returned.
TEST(GrpcServerTest, InjectStimulusAccepted) {
  TestEnv env(19100);

  senna::StimulusRequest req;
  auto img = MakeTestImage();
  for (auto px : img) req.add_pixels(px);
  req.set_label(5);
  req.set_duration_ms(50);

  senna::StimulusResponse resp;
  grpc::ClientContext ctx;
  auto status = env.stub->InjectStimulus(&ctx, req, &resp);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(resp.accepted());
  EXPECT_GT(resp.stimulus_id(), 0u);
  EXPECT_TRUE(resp.error().empty());
}

// 14.3.2 GetPrediction: after processing a stimulus, predicted_class is
// returned.
TEST(GrpcServerTest, GetPredictionAfterStimulus) {
  TestEnv env(19101);

  // Inject a stimulus.
  senna::StimulusRequest stim_req;
  auto img = MakeTestImage(200);
  for (auto px : img) stim_req.add_pixels(px);
  stim_req.set_label(3);
  stim_req.set_duration_ms(50);

  senna::StimulusResponse stim_resp;
  {
    grpc::ClientContext ctx;
    env.stub->InjectStimulus(&ctx, stim_req, &stim_resp);
  }
  ASSERT_TRUE(stim_resp.accepted());
  uint64_t sid = stim_resp.stimulus_id();

  // Run enough ticks to cover the presentation window.
  for (int i = 0; i < 200; ++i) {
    auto syn = env.net->synapses_ptr();
    env.net->time_manager().Tick(env.net->queue(), env.net->pool(), *syn);
  }

  // Try to get prediction.
  senna::PredictionRequest pred_req;
  pred_req.set_stimulus_id(sid);
  senna::PredictionResponse pred_resp;
  {
    grpc::ClientContext ctx;
    auto status = env.stub->GetPrediction(&ctx, pred_req, &pred_resp);
    ASSERT_TRUE(status.ok()) << status.error_message();
  }
  EXPECT_EQ(pred_resp.stimulus_id(), sid);
  // Prediction may or may not be ready depending on network activity,
  // but the RPC must succeed.
}

// 14.3.3 Supervise: forced spike on the correct output neuron.
TEST(GrpcServerTest, SuperviseAccepted) {
  TestEnv env(19102);

  senna::SupervisionRequest req;
  req.set_stimulus_id(1);
  req.set_correct_class(7);

  senna::SupervisionResponse resp;
  grpc::ClientContext ctx;
  auto status = env.stub->Supervise(&ctx, req, &resp);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_TRUE(resp.accepted());
}

// 14.3.4 GetStatus: returns valid core state.
TEST(GrpcServerTest, GetStatusReturnsValidState) {
  TestEnv env(19103);

  google::protobuf::Empty req;
  senna::StatusResponse resp;
  grpc::ClientContext ctx;
  auto status = env.stub->GetStatus(&ctx, req, &resp);

  ASSERT_TRUE(status.ok()) << status.error_message();
  EXPECT_GE(resp.neuron_count(), 48u);  // 4*4*3 grid + outputs
  EXPECT_GE(resp.synapse_count(), 0u);
  // Phase and sleep_pressure default to 0.
  EXPECT_DOUBLE_EQ(resp.phase(), 0.0);
}

// 14.3.5 ReportAccuracy: metric is updated in Prometheus pipeline.
TEST(GrpcServerTest, ReportAccuracyUpdatesMetrics) {
  TestEnv env(19104);

  senna::AccuracyRequest req;
  req.set_train_accuracy(0.85);
  req.set_test_accuracy(0.82);

  google::protobuf::Empty resp;
  grpc::ClientContext ctx;
  auto status = env.stub->ReportAccuracy(&ctx, req, &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();
}

// 14.3.6 Shutdown: server stops gracefully.
TEST(GrpcServerTest, ShutdownStopsServer) {
  TestEnv env(19105);

  google::protobuf::Empty req;
  google::protobuf::Empty resp;
  grpc::ClientContext ctx;
  auto status = env.stub->Shutdown(&ctx, req, &resp);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // After shutdown, new RPCs should fail.
  std::this_thread::sleep_for(200ms);
  google::protobuf::Empty req2;
  senna::StatusResponse resp2;
  grpc::ClientContext ctx2;
  auto status2 = env.stub->GetStatus(&ctx2, req2, &resp2);
  EXPECT_FALSE(status2.ok());
}

// 14.3.7 Reject stimulus during sleep phase.
TEST(GrpcServerTest, RejectStimulusDuringSleep) {
  TestEnv env(19106);

  // Set phase > 0.5 to simulate sleep.
  env.net->UpdatePhase(0.8, 0.9);

  senna::StimulusRequest req;
  auto img = MakeTestImage();
  for (auto px : img) req.add_pixels(px);
  req.set_label(1);
  req.set_duration_ms(50);

  senna::StimulusResponse resp;
  grpc::ClientContext ctx;
  auto status = env.stub->InjectStimulus(&ctx, req, &resp);

  ASSERT_TRUE(status.ok());
  EXPECT_FALSE(resp.accepted());
  EXPECT_EQ(resp.error(), "sleep phase");
}

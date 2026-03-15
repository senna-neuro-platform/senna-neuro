#include "core/interfaces/grpc_server.hpp"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "senna.grpc.pb.h"
#include "senna.pb.h"

namespace senna::interfaces {

// ---------------------------------------------------------------------------
// Service implementation
// ---------------------------------------------------------------------------

class SennaCoreServiceImpl final : public senna::SennaCore::Service {
 public:
  explicit SennaCoreServiceImpl(GrpcServer* owner) : owner_(owner) {}

  ::grpc::Status InjectStimulus(::grpc::ServerContext* /*ctx*/,
                                const senna::StimulusRequest* req,
                                senna::StimulusResponse* resp) override {
    if (owner_->shutting_down()) {
      resp->set_accepted(false);
      resp->set_error("shutting down");
      return ::grpc::Status::OK;
    }
    // Reject during sleep phase (phase > 0.5).
    if (owner_->net()->phase() > 0.5) {
      resp->set_accepted(false);
      resp->set_error("sleep phase");
      return ::grpc::Status::OK;
    }
    std::vector<uint32_t> pixels(req->pixels().begin(), req->pixels().end());
    uint32_t duration = req->duration_ms() > 0 ? req->duration_ms() : 50;
    uint64_t sid = owner_->InjectStimulus(pixels, req->label(), duration);
    resp->set_accepted(true);
    resp->set_stimulus_id(sid);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetPrediction(::grpc::ServerContext* /*ctx*/,
                               const senna::PredictionRequest* req,
                               senna::PredictionResponse* resp) override {
    owner_->TryDecode(req->stimulus_id());
    auto* rec = owner_->FindStimulus(req->stimulus_id());
    if (rec == nullptr) {
      resp->set_ready(false);
      return ::grpc::Status::OK;
    }
    resp->set_stimulus_id(rec->id);
    resp->set_ready(rec->ready);
    if (rec->ready) {
      resp->set_predicted_class(rec->predicted_class.value_or(-1));
      resp->set_confidence(rec->confidence);
      resp->set_latency_ms(rec->latency_ms);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status Supervise(::grpc::ServerContext* /*ctx*/,
                           const senna::SupervisionRequest* req,
                           senna::SupervisionResponse* resp) override {
    int cls = req->correct_class();
    const auto& output_ids = owner_->net()->output_ids();
    if (cls < 0 || cls >= static_cast<int>(output_ids.size())) {
      resp->set_accepted(false);
      resp->set_error("invalid class");
      return ::grpc::Status::OK;
    }
    int32_t neuron_id = output_ids[cls];
    float t = owner_->net()->time_manager().time();
    owner_->net()->InjectSpike(neuron_id, t, 5.0F);
    resp->set_accepted(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetStatus(::grpc::ServerContext* /*ctx*/,
                           const google::protobuf::Empty* /*req*/,
                           senna::StatusResponse* resp) override {
    auto* net = owner_->net();
    resp->set_phase(net->phase());
    resp->set_sleep_pressure(net->sleep_pressure());
    resp->set_virtual_time_ms(
        static_cast<uint64_t>(net->time_manager().time()));
    resp->set_neuron_count(net->pool().size());
    resp->set_synapse_count(net->synapses_ptr()->synapse_count());
    return ::grpc::Status::OK;
  }

  ::grpc::Status Configure(::grpc::ServerContext* /*ctx*/,
                           const senna::ConfigRequest* /*req*/,
                           senna::ConfigResponse* resp) override {
    // Placeholder: runtime config hot-reload can be extended later.
    resp->set_applied(true);
    return ::grpc::Status::OK;
  }

  ::grpc::Status StreamPredictions(
      ::grpc::ServerContext* ctx, const google::protobuf::Empty* /*req*/,
      ::grpc::ServerWriter<senna::PredictionEvent>* writer) override {
    while (!ctx->IsCancelled() && !owner_->shutting_down()) {
      GrpcServer::PredictionNotification note{};
      {
        std::unique_lock lock(owner_->stream_mutex_);
        owner_->stream_cv_.wait_for(lock, std::chrono::milliseconds(100), [&] {
          return !owner_->pending_predictions_.empty() || ctx->IsCancelled() ||
                 owner_->shutting_down();
        });
        if (owner_->pending_predictions_.empty()) {
          continue;
        }
        note = owner_->pending_predictions_.front();
        owner_->pending_predictions_.pop_front();
      }
      senna::PredictionEvent evt;
      evt.set_stimulus_id(note.stimulus_id);
      evt.set_predicted_class(note.predicted_class);
      evt.set_confidence(note.confidence);
      evt.set_latency_ms(note.latency_ms);
      if (!writer->Write(evt)) {
        break;
      }
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status ReportAccuracy(::grpc::ServerContext* /*ctx*/,
                                const senna::AccuracyRequest* req,
                                google::protobuf::Empty* /*resp*/) override {
    if (owner_->metrics() != nullptr) {
      owner_->metrics()->RecordAccuracy(req->train_accuracy(),
                                        req->test_accuracy());
    }
    owner_->net()->UpdateAccuracy(req->train_accuracy(), req->test_accuracy());
    return ::grpc::Status::OK;
  }

  ::grpc::Status Shutdown(::grpc::ServerContext* /*ctx*/,
                          const google::protobuf::Empty* /*req*/,
                          google::protobuf::Empty* /*resp*/) override {
    // Stop asynchronously so the RPC can return OK before shutdown.
    std::thread([owner = owner_]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      owner->Stop();
    }).detach();
    return ::grpc::Status::OK;
  }

 private:
  // Allow access to private notification fields.
  friend class GrpcServer;
  GrpcServer* owner_;
};

// ---------------------------------------------------------------------------
// GrpcServer implementation
// ---------------------------------------------------------------------------

GrpcServer::GrpcServer(int port, network::Network* net,
                       observability::MetricsCollector* metrics)
    : port_(port), net_(net), metrics_(metrics) {}

GrpcServer::~GrpcServer() { Stop(); }

void GrpcServer::Start() {
  stop_.store(false);
  server_thread_ = std::thread([this]() {
    std::string addr = "0.0.0.0:" + std::to_string(port_);
    auto service = std::make_unique<SennaCoreServiceImpl>(this);

    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, ::grpc::InsecureServerCredentials());
    builder.RegisterService(service.get());
    server_ = builder.BuildAndStart();
    if (server_) {
      server_->Wait();
    }
  });
}

void GrpcServer::Stop() {
  if (stop_.exchange(true)) {
    return;
  }
  // Wake any blocked StreamPredictions.
  stream_cv_.notify_all();
  if (server_) {
    server_->Shutdown(std::chrono::system_clock::now() +
                      std::chrono::seconds(2));
  }
  if (server_thread_.joinable()) {
    server_thread_.join();
  }
}

uint64_t GrpcServer::InjectStimulus(const std::vector<uint32_t>& pixels,
                                    int32_t label, uint32_t duration_ms) {
  float t_start = net_->time_manager().time();
  // Convert uint32 pixels to uint8 for EncodeImage.
  std::vector<uint8_t> img(pixels.size());
  for (size_t i = 0; i < pixels.size(); ++i) {
    img[i] = static_cast<uint8_t>(std::min(pixels[i], 255U));
  }
  net_->EncodeImage(img, t_start);

  std::scoped_lock lock(stimuli_mutex_);
  uint64_t sid = next_stimulus_id_++;
  stimuli_.push_back(StimulusRecord{
      .id = sid,
      .label = label,
      .inject_time_ms = t_start,
      .duration_ms = static_cast<float>(duration_ms),
  });
  return sid;
}

StimulusRecord* GrpcServer::FindStimulus(uint64_t id) {
  std::scoped_lock lock(stimuli_mutex_);
  for (auto& r : stimuli_) {
    if (r.id == id) {
      return &r;
    }
  }
  return nullptr;
}

void GrpcServer::TryDecode(uint64_t stimulus_id) {
  std::scoped_lock lock(stimuli_mutex_);
  for (auto& r : stimuli_) {
    if (r.id != stimulus_id || r.ready) {
      continue;
    }
    float t_now = net_->time_manager().time();
    float window_end = r.inject_time_ms + r.duration_ms;
    if (t_now < window_end) {
      continue;  // not enough time elapsed
    }

    // Create a temporary decoder and replay recent spikes.
    decoding::FirstSpikeDecoder decoder(net_->output_ids(), r.duration_ms);
    decoder.SetStartTime(r.inject_time_ms);
    auto spikes = net_->time_manager().LastSpikesCopy();
    for (auto& [nid, t] : spikes) {
      decoder.Observe(nid, t);
    }
    auto result = decoder.ResultWithTimeout(t_now);
    if (result.has_value()) {
      r.predicted_class = result;
      r.confidence =
          1.0 / std::max(1.0, static_cast<double>(t_now - r.inject_time_ms));
      r.latency_ms = static_cast<uint32_t>(t_now - r.inject_time_ms);
      r.ready = true;
      NotifyPrediction(r.id, r.predicted_class.value_or(-1), r.confidence,
                       r.latency_ms);
    }
  }
}

void GrpcServer::NotifyPrediction(uint64_t stimulus_id, int predicted_class,
                                  double confidence, uint32_t latency_ms) {
  std::scoped_lock lock(stream_mutex_);
  pending_predictions_.push_back(PredictionNotification{
      stimulus_id, predicted_class, confidence, latency_ms});
  stream_cv_.notify_all();
}

}  // namespace senna::interfaces

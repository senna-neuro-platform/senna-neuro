#include "core/network/network_builder.hpp"

#include "core/plasticity/structural.hpp"

namespace senna::network {

static std::vector<int32_t> BuildOutputIds(const spatial::ZonedLattice& lattice,
                                           int num_outputs) {
  std::vector<int32_t> ids;
  ids.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ids.push_back(lattice.OutputNeuron(i));
  }
  return ids;
}

Network::Network(const NetworkConfig& config,
                 observability::MetricsCollector* metrics)
    : config_(config),
      lattice_(config.width, config.height, config.depth, config.density,
               config.seed, config.num_outputs),
      neighbors_(lattice_, config.neighbor_radius, 0),
      pool_(lattice_, config.lif_params, config.excitatory_ratio, config.seed),
      output_ids_(BuildOutputIds(lattice_, config.num_outputs)),
      synapses_ptr_atomic_(std::make_shared<synaptic::SynapseIndex>(
          lattice_, neighbors_, pool_, output_ids_, config.synapse_params,
          config.seed)),
      encoder_(config.encoder_params, config.dt, config.seed),
      time_manager_(config.dt, config.homeostasis, config.seed),
      metrics_(metrics) {
  time_manager_.attach_pool(&pool_);
  time_manager_.attach_metrics(metrics_);
  synapses_cache_ = synapses_ptr_atomic_.load();
  structural_worker_ = std::make_unique<plasticity::StructuralWorker>(
      lattice_, neighbors_, pool_, synapses_ptr_atomic_, config.synapse_params,
      config.structural, config.homeostasis.target_rate_hz, metrics_);
  structural_worker_->Start();
}

void Network::InjectSpike(int32_t neuron_id, float time, float value) {
  queue_.Push({.target_id = neuron_id,
               .source_id = -1,
               .arrival_time = time,
               .value = value});
}

void Network::InjectSensory(int x, int y, float time, float value) {
  auto id = lattice_.SensoryNeuron(x, y);
  if (id != spatial::kEmptyVoxel) {
    InjectSpike(id, time, value);
  }
}

void Network::EncodeImage(std::span<const uint8_t> image, float t_start) {
  encoder_.Encode(image, lattice_, queue_, t_start);
}

void Network::UpdatePhase(double phase, double sleep_pressure) {
  if (metrics_ != nullptr) {
    metrics_->RecordPhase(phase, sleep_pressure);
  }
}

void Network::UpdateAccuracy(double train, double test) {
  if (metrics_ != nullptr) {
    metrics_->RecordAccuracy(train, test);
  }
}

}  // namespace senna::network

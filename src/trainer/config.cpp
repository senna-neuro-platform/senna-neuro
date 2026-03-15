#include "trainer/config.hpp"

#include <yaml-cpp/yaml.h>

namespace senna::trainer {

namespace {
template <typename T>
T Get(const YAML::Node& node, const char* key, const T& def) {
  if (!node || !node[key]) {
    return def;
  }
  try {
    return node[key].as<T>();
  } catch (...) {
    return def;
  }
}
}  // namespace

TrainerConfig LoadTrainerConfig(const std::string& path) {
  TrainerConfig cfg;
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (...) {
    return cfg;
  }

  // Connection settings from existing trainer section.
  if (auto t = root["trainer"]) {
    cfg.core_host = Get<std::string>(t, "host", cfg.core_host);
    cfg.core_port = Get<int>(t, "port", cfg.core_port);
    cfg.epochs = Get<int>(t, "epochs", cfg.epochs);
    cfg.presentation_ms = Get<int>(t, "presentation_ms", cfg.presentation_ms);
    cfg.inter_stimulus_ms =
        Get<int>(t, "inter_stimulus_ms", cfg.inter_stimulus_ms);
    cfg.prediction_timeout_ms =
        Get<int>(t, "prediction_timeout_ms", cfg.prediction_timeout_ms);
    cfg.prediction_poll_ms =
        Get<int>(t, "prediction_poll_ms", cfg.prediction_poll_ms);
    cfg.max_train_samples =
        Get<int>(t, "max_train_samples", cfg.max_train_samples);
    cfg.max_test_samples =
        Get<int>(t, "max_test_samples", cfg.max_test_samples);
  }

  // Fall back to ports.grpc if trainer port not set.
  if (auto t = root["trainer"]; !t || !t["port"]) {
    if (auto ports = root["ports"]) {
      cfg.core_port = Get<int>(ports, "grpc", cfg.core_port);
    }
  }

  // MNIST paths from data section (optional).
  if (auto data = root["data"]) {
    cfg.train_images = Get<std::string>(data, "train_images", cfg.train_images);
    cfg.train_labels = Get<std::string>(data, "train_labels", cfg.train_labels);
    cfg.test_images = Get<std::string>(data, "test_images", cfg.test_images);
    cfg.test_labels = Get<std::string>(data, "test_labels", cfg.test_labels);
  }

  // Use encoder presentation_ms as default if not overridden.
  if (auto t = root["trainer"]; !t || !t["presentation_ms"]) {
    if (auto enc = root["encoder"]) {
      cfg.presentation_ms =
          Get<int>(enc, "presentation_ms", cfg.presentation_ms);
    }
  }

  return cfg;
}

}  // namespace senna::trainer

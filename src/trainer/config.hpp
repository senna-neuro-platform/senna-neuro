#pragma once

#include <string>

namespace senna::trainer {

// Trainer-specific configuration loaded from YAML.
struct TrainerConfig {
  // Core connection.
  std::string core_host = "senna-core";
  int core_port = 50051;

  // Training parameters.
  int epochs = 1;
  int presentation_ms = 50;
  int inter_stimulus_ms = 10;
  int prediction_timeout_ms = 500;
  int prediction_poll_ms = 5;

  // MNIST data paths.
  std::string train_images = "data/train-images-idx3-ubyte";
  std::string train_labels = "data/train-labels-idx1-ubyte";
  std::string test_images = "data/t10k-images-idx3-ubyte";
  std::string test_labels = "data/t10k-labels-idx1-ubyte";

  // Limit samples per epoch (0 = all).
  int max_train_samples = 0;
  int max_test_samples = 0;
};

// Load trainer config from YAML. Falls back to defaults for missing fields.
TrainerConfig LoadTrainerConfig(const std::string& path);

}  // namespace senna::trainer

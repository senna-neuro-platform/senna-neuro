#include <cstdlib>
#include <iostream>
#include <string>

#include "trainer/config.hpp"
#include "trainer/mnist_loader.hpp"
#include "trainer/training_pipeline.hpp"

namespace {

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " <command> [options]\n"
      << "\nCommands:\n"
      << "  train    Run training epochs on MNIST\n"
      << "  test     Evaluate on test set only\n"
      << "  bench    Benchmark: train + test + report\n"
      << "\nOptions:\n"
      << "  --config <path>   Config file (default: configs/default.yaml)\n"
      << "  --epochs <n>      Override number of epochs\n"
      << "  --max-train <n>   Limit training samples per epoch\n"
      << "  --max-test <n>    Limit test samples\n"
      << "  --host <host>     Core host (default: from config)\n"
      << "  --port <port>     Core port (default: from config)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  std::string command = argv[1];
  if (command != "train" && command != "test" && command != "bench") {
    std::cerr << "Unknown command: " << command << "\n";
    PrintUsage(argv[0]);
    return 1;
  }

  // Parse options.
  std::string config_path = "configs/default.yaml";
  int epochs_override = -1;
  int max_train = -1;
  int max_test = -1;
  std::string host_override;
  int port_override = -1;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--epochs" && i + 1 < argc) {
      epochs_override = std::atoi(argv[++i]);
    } else if (arg == "--max-train" && i + 1 < argc) {
      max_train = std::atoi(argv[++i]);
    } else if (arg == "--max-test" && i + 1 < argc) {
      max_test = std::atoi(argv[++i]);
    } else if (arg == "--host" && i + 1 < argc) {
      host_override = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      port_override = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }
  }

  // Load config.
  auto cfg = senna::trainer::LoadTrainerConfig(config_path);

  // Apply env overrides.
  if (const char* env =
          std::getenv("SENNA_CORE_HOST")) {  // NOLINT(concurrency-mt-unsafe)
    cfg.core_host = env;
  }
  if (const char* env =
          std::getenv("SENNA_CORE_PORT")) {  // NOLINT(concurrency-mt-unsafe)
    cfg.core_port = std::atoi(env);
  }

  // Apply CLI overrides.
  if (!host_override.empty()) {
    cfg.core_host = host_override;
  }
  if (port_override > 0) {
    cfg.core_port = port_override;
  }
  if (epochs_override > 0) {
    cfg.epochs = epochs_override;
  }
  if (max_train > 0) {
    cfg.max_train_samples = max_train;
  }
  if (max_test > 0) {
    cfg.max_test_samples = max_test;
  }

  // Load MNIST data.
  senna::trainer::MnistLoader train_data;
  senna::trainer::MnistLoader test_data;

  if (command == "train" || command == "bench") {
    std::cout << "Loading training data...\n";
    if (!train_data.Load(cfg.train_images, cfg.train_labels)) {
      std::cerr << "Failed to load training data\n";
      return 1;
    }
    std::cout << "Loaded " << train_data.size() << " training samples\n";
  }

  if (command == "test" || command == "train" || command == "bench") {
    std::cout << "Loading test data...\n";
    if (!test_data.Load(cfg.test_images, cfg.test_labels)) {
      std::cerr << "Failed to load test data\n";
      return 1;
    }
    std::cout << "Loaded " << test_data.size() << " test samples\n";
  }

  // Connect to core.
  senna::trainer::TrainingPipeline pipeline(cfg);
  pipeline.set_epoch_callback([](const senna::trainer::EpochResult& r) {
    std::cout << (r.is_test ? "[TEST]" : "[TRAIN]") << " epoch=" << r.epoch
              << " correct=" << r.correct << "/" << r.total
              << " acc=" << r.accuracy << "\n";
  });

  if (!pipeline.Connect()) {
    std::cerr << "Failed to connect to senna-core at " << cfg.core_host << ":"
              << cfg.core_port << "\n";
    return 1;
  }

  // Execute command.
  if (command == "train" || command == "bench") {
    pipeline.Train(train_data, test_data);
  } else if (command == "test") {
    auto result = pipeline.Test(test_data);
    std::cout << "Test accuracy: " << result.accuracy << " (" << result.correct
              << "/" << result.total << ")\n";
  }

  return 0;
}

#include "trainer/config.hpp"

#include <gtest/gtest.h>

#include <fstream>

// 15.5.5 Load trainer config from default.yaml.
TEST(TrainerConfigTest, LoadsFromDefaultYaml) {
  auto cfg = senna::trainer::LoadTrainerConfig("configs/default.yaml");

  EXPECT_EQ(cfg.core_host, "senna-core");
  EXPECT_EQ(cfg.core_port, 50051);
  EXPECT_EQ(cfg.presentation_ms, 50);
}

// 15.5.6 Missing file returns defaults.
TEST(TrainerConfigTest, MissingFileUsesDefaults) {
  auto cfg = senna::trainer::LoadTrainerConfig("/tmp/nonexistent_config.yaml");

  EXPECT_EQ(cfg.core_host, "senna-core");
  EXPECT_EQ(cfg.core_port, 50051);
  EXPECT_EQ(cfg.epochs, 1);
  EXPECT_EQ(cfg.prediction_timeout_ms, 500);
}

// 15.5.7 Custom trainer section is read.
TEST(TrainerConfigTest, ReadsCustomTrainerSection) {
  const char* yaml = R"(
trainer:
  host: localhost
  port: 9999
  epochs: 5
  inter_stimulus_ms: 20
  prediction_timeout_ms: 1000
  max_train_samples: 100
  max_test_samples: 50
)";
  std::ofstream("/tmp/test_trainer_cfg.yaml") << yaml;

  auto cfg = senna::trainer::LoadTrainerConfig("/tmp/test_trainer_cfg.yaml");
  EXPECT_EQ(cfg.core_host, "localhost");
  EXPECT_EQ(cfg.core_port, 9999);
  EXPECT_EQ(cfg.epochs, 5);
  EXPECT_EQ(cfg.inter_stimulus_ms, 20);
  EXPECT_EQ(cfg.prediction_timeout_ms, 1000);
  EXPECT_EQ(cfg.max_train_samples, 100);
  EXPECT_EQ(cfg.max_test_samples, 50);
}

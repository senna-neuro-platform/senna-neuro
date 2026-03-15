#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace senna::trainer {

// A single MNIST sample: 28x28 pixel values and label (0-9).
struct MnistSample {
  std::vector<uint8_t> pixels;  // 784 elements
  uint8_t label{0};
};

// Loads MNIST IDX binary files (train-images, train-labels, etc.).
class MnistLoader {
 public:
  // Load images and labels from their respective IDX files.
  // Returns false on error (file not found, bad magic, size mismatch).
  bool Load(const std::string& images_path, const std::string& labels_path);

  const std::vector<MnistSample>& samples() const { return samples_; }
  size_t size() const { return samples_.size(); }
  const MnistSample& operator[](size_t i) const { return samples_[i]; }

 private:
  std::vector<MnistSample> samples_;
};

}  // namespace senna::trainer

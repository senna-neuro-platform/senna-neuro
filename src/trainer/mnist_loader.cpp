#include "trainer/mnist_loader.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>

namespace senna::trainer {

namespace {

// Read a big-endian 32-bit integer from an input stream.
uint32_t ReadU32BE(std::ifstream& in) {
  std::array<char, 4> buf{};
  in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  return (static_cast<uint32_t>(static_cast<unsigned char>(buf[0])) << 24) |
         (static_cast<uint32_t>(static_cast<unsigned char>(buf[1])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(buf[2])) << 8) |
         static_cast<uint32_t>(static_cast<unsigned char>(buf[3]));
}

}  // namespace

bool MnistLoader::Load(const std::string& images_path,
                       const std::string& labels_path) {
  samples_.clear();

  // --- Read images ---
  std::ifstream img_file(images_path, std::ios::binary);
  if (!img_file.is_open()) {
    std::cerr << "MnistLoader: cannot open " << images_path << "\n";
    return false;
  }

  uint32_t img_magic = ReadU32BE(img_file);
  if (img_magic != 0x00000803) {
    std::cerr << "MnistLoader: bad image magic 0x" << std::hex << img_magic
              << "\n";
    return false;
  }

  uint32_t num_images = ReadU32BE(img_file);
  uint32_t rows = ReadU32BE(img_file);
  uint32_t cols = ReadU32BE(img_file);
  uint32_t pixels_per_image = rows * cols;

  // --- Read labels ---
  std::ifstream lbl_file(labels_path, std::ios::binary);
  if (!lbl_file.is_open()) {
    std::cerr << "MnistLoader: cannot open " << labels_path << "\n";
    return false;
  }

  uint32_t lbl_magic = ReadU32BE(lbl_file);
  if (lbl_magic != 0x00000801) {
    std::cerr << "MnistLoader: bad label magic 0x" << std::hex << lbl_magic
              << "\n";
    return false;
  }

  uint32_t num_labels = ReadU32BE(lbl_file);
  if (num_labels != num_images) {
    std::cerr << "MnistLoader: image/label count mismatch (" << num_images
              << " vs " << num_labels << ")\n";
    return false;
  }

  // --- Read all samples ---
  samples_.resize(num_images);
  for (uint32_t i = 0; i < num_images; ++i) {
    samples_[i].pixels.resize(pixels_per_image);
    img_file.read(reinterpret_cast<char*>(samples_[i].pixels.data()),  // NOLINT
                  static_cast<std::streamsize>(pixels_per_image));
    uint8_t label{};
    lbl_file.read(reinterpret_cast<char*>(&label), 1);  // NOLINT
    samples_[i].label = label;
  }

  if (!img_file.good() || !lbl_file.good()) {
    std::cerr << "MnistLoader: read error (truncated file?)\n";
    samples_.clear();
    return false;
  }

  return true;
}

}  // namespace senna::trainer

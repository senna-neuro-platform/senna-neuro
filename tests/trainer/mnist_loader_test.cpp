#include "trainer/mnist_loader.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Write a big-endian 32-bit integer to a stream.
void WriteU32BE(std::ofstream& out, uint32_t val) {
  uint8_t buf[4] = {
      static_cast<uint8_t>((val >> 24) & 0xFF),
      static_cast<uint8_t>((val >> 16) & 0xFF),
      static_cast<uint8_t>((val >> 8) & 0xFF),
      static_cast<uint8_t>(val & 0xFF),
  };
  out.write(reinterpret_cast<const char*>(buf), 4);
}

// Create a synthetic MNIST images file (IDX3 format).
std::string CreateTestImages(const std::string& path, int count, int rows,
                             int cols, uint8_t fill = 128) {
  std::ofstream out(path, std::ios::binary);
  WriteU32BE(out, 0x00000803);  // magic
  WriteU32BE(out, count);
  WriteU32BE(out, rows);
  WriteU32BE(out, cols);
  std::vector<uint8_t> pixels(rows * cols, fill);
  for (int i = 0; i < count; ++i) {
    // Vary first pixel so images aren't identical.
    pixels[0] = static_cast<uint8_t>(i % 256);
    out.write(reinterpret_cast<const char*>(pixels.data()),
              static_cast<std::streamsize>(pixels.size()));
  }
  out.close();
  return path;
}

// Create a synthetic MNIST labels file (IDX1 format).
std::string CreateTestLabels(const std::string& path, int count) {
  std::ofstream out(path, std::ios::binary);
  WriteU32BE(out, 0x00000801);  // magic
  WriteU32BE(out, count);
  for (int i = 0; i < count; ++i) {
    uint8_t label = static_cast<uint8_t>(i % 10);
    out.write(reinterpret_cast<const char*>(&label), 1);
  }
  out.close();
  return path;
}

}  // namespace

// 15.5.1 Load synthetic MNIST data and verify contents.
TEST(MnistLoaderTest, LoadsSyntheticData) {
  const int count = 20;
  auto img_path = CreateTestImages("/tmp/test_images.idx3", count, 28, 28);
  auto lbl_path = CreateTestLabels("/tmp/test_labels.idx1", count);

  senna::trainer::MnistLoader loader;
  ASSERT_TRUE(loader.Load(img_path, lbl_path));
  EXPECT_EQ(loader.size(), count);

  for (size_t i = 0; i < loader.size(); ++i) {
    EXPECT_EQ(loader[i].pixels.size(), 784u);
    EXPECT_EQ(loader[i].label, i % 10);
    // First pixel should be i % 256.
    EXPECT_EQ(loader[i].pixels[0], static_cast<uint8_t>(i % 256));
  }
}

// 15.5.2 Bad image magic returns false.
TEST(MnistLoaderTest, RejectsBadMagic) {
  // Write an images file with wrong magic.
  {
    std::ofstream out("/tmp/bad_magic.idx3", std::ios::binary);
    WriteU32BE(out, 0xDEADBEEF);
    WriteU32BE(out, 1);
    WriteU32BE(out, 28);
    WriteU32BE(out, 28);
  }
  auto lbl_path = CreateTestLabels("/tmp/bad_magic_labels.idx1", 1);

  senna::trainer::MnistLoader loader;
  EXPECT_FALSE(loader.Load("/tmp/bad_magic.idx3", lbl_path));
  EXPECT_EQ(loader.size(), 0u);
}

// 15.5.3 Count mismatch between images and labels.
TEST(MnistLoaderTest, RejectsCountMismatch) {
  auto img_path = CreateTestImages("/tmp/mismatch_images.idx3", 10, 28, 28);
  auto lbl_path = CreateTestLabels("/tmp/mismatch_labels.idx1", 5);

  senna::trainer::MnistLoader loader;
  EXPECT_FALSE(loader.Load(img_path, lbl_path));
}

// 15.5.4 Missing file returns false.
TEST(MnistLoaderTest, MissingFileReturnsFalse) {
  senna::trainer::MnistLoader loader;
  EXPECT_FALSE(loader.Load("/tmp/nonexistent_images.idx3",
                           "/tmp/nonexistent_labels.idx1"));
}

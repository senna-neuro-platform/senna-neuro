#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <vector>

namespace senna::decoding {

// First-spike decoder: returns the class index (position in output_ids)
// of the first output neuron that fires. If no output neuron fires,
// Result() is std::nullopt.
class FirstSpikeDecoder {
 public:
  explicit FirstSpikeDecoder(std::vector<int32_t> output_ids,
                             float window_ms = 50.0F, uint64_t seed = 42);

  // Observe a spike; neuron_id is a global neuron index, t is spike time.
  void Observe(int32_t neuron_id, float t);

  // Reset decoder state for a new trial.
  void Reset(float t_start_ms = 0.0F);

  // Resolve pending simultaneous candidates (if any) using RNG.
  void Finalize(float t_now);

  // Returns the winning class index if any (ignores timeout). Finalizes pending
  // candidates if needed.
  std::optional<int> Result();

  // Returns winner if decided; otherwise returns empty when the time window
  // [start_time, start_time + window_ms] has elapsed.
  std::optional<int> ResultWithTimeout(float t_now);

  void SetWindow(float window_ms) { window_ms_ = window_ms; }
  void SetStartTime(float t_start_ms) {
    start_time_ms_ = t_start_ms;
    window_expired_ = false;
    winner_.reset();
    candidates_.clear();
    earliest_time_ = std::numeric_limits<float>::max();
  }
  void SetSeed(uint64_t seed) { rng_.seed(seed); }

 private:
  std::vector<int32_t> output_ids_;
  std::optional<int> winner_;
  float winner_time_ = 0.0F;
  float window_ms_;
  float start_time_ms_ = 0.0F;
  bool window_expired_ = false;
  float earliest_time_ = std::numeric_limits<float>::max();
  std::vector<int> candidates_;  // indices into output_ids_
  std::mt19937_64 rng_;
};

}  // namespace senna::decoding

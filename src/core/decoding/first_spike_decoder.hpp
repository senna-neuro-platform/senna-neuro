#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace senna::decoding {

// First-spike decoder: returns the class index (position in output_ids)
// of the first output neuron that fires. If no output neuron fires,
// Result() is std::nullopt.
class FirstSpikeDecoder {
 public:
  explicit FirstSpikeDecoder(std::vector<int32_t> output_ids,
                             float window_ms = 50.0f);

  // Observe a spike; neuron_id is a global neuron index, t is spike time.
  void Observe(int32_t neuron_id, float t);

  // Reset decoder state for a new trial.
  void Reset(float t_start_ms = 0.0f);

  // Returns the winning class index if any (ignores timeout).
  std::optional<int> Result() const { return winner_; }

  // Returns winner if decided; otherwise returns empty when the time window
  // [start_time, start_time + window_ms] has elapsed.
  std::optional<int> ResultWithTimeout(float t_now) const;

  void SetWindow(float window_ms) { window_ms_ = window_ms; }
  void SetStartTime(float t_start_ms) {
    start_time_ms_ = t_start_ms;
    window_expired_ = false;
    winner_.reset();
  }

 private:
  std::vector<int32_t> output_ids_;
  std::optional<int> winner_;
  float winner_time_ = 0.0f;
  float window_ms_;
  float start_time_ms_ = 0.0f;
  bool window_expired_ = false;
};

}  // namespace senna::decoding

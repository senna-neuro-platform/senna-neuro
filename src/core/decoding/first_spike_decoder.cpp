#include "core/decoding/first_spike_decoder.hpp"

#include <algorithm>

namespace senna::decoding {

FirstSpikeDecoder::FirstSpikeDecoder(std::vector<int32_t> output_ids,
                                     float window_ms)
    : output_ids_(std::move(output_ids)), window_ms_(window_ms) {}

void FirstSpikeDecoder::Reset(float t_start_ms) {
  winner_.reset();
  winner_time_ = 0.0f;
  window_expired_ = false;
  start_time_ms_ = t_start_ms;
}

void FirstSpikeDecoder::Observe(int32_t neuron_id, float t) {
  if (window_expired_ || winner_.has_value()) return;

  // Window elapsed without a winner — lock in "undefined".
  if ((t - start_time_ms_) > window_ms_) {
    window_expired_ = true;
    return;
  }

  auto it = std::find(output_ids_.begin(), output_ids_.end(), neuron_id);
  if (it == output_ids_.end()) return;

  int idx = static_cast<int>(std::distance(output_ids_.begin(), it));
  winner_ = idx;
  winner_time_ = t;
}

std::optional<int> FirstSpikeDecoder::ResultWithTimeout(float t_now) const {
  if (winner_.has_value()) return winner_;
  if ((t_now - start_time_ms_) >= window_ms_) return std::nullopt;
  return std::nullopt;
}

}  // namespace senna::decoding

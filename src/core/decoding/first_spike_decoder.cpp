#include "core/decoding/first_spike_decoder.hpp"

#include <algorithm>
#include <cmath>

namespace senna::decoding {

FirstSpikeDecoder::FirstSpikeDecoder(std::vector<int32_t> output_ids,
                                     float window_ms, uint64_t seed)
    : output_ids_(std::move(output_ids)), window_ms_(window_ms), rng_(seed) {}

void FirstSpikeDecoder::Reset(float t_start_ms) {
  winner_.reset();
  winner_time_ = 0.0F;
  window_expired_ = false;
  start_time_ms_ = t_start_ms;
  candidates_.clear();
  earliest_time_ = std::numeric_limits<float>::max();
}

void FirstSpikeDecoder::Observe(int32_t neuron_id, float t) {
  if (window_expired_ || winner_.has_value()) {
    return;
  }

  // Window elapsed without a winner - mark expired.
  if ((t - start_time_ms_) > window_ms_) {
    window_expired_ = true;
    return;
  }

  auto it = std::find(output_ids_.begin(), output_ids_.end(), neuron_id);
  if (it == output_ids_.end()) {
    return;
  }

  int idx = static_cast<int>(std::distance(output_ids_.begin(), it));
  constexpr float kEps = 1e-6F;

  if (candidates_.empty()) {
    earliest_time_ = t;
    candidates_.push_back(idx);
    return;
  }

  if (std::fabs(t - earliest_time_) < kEps) {
    candidates_.push_back(idx);
    return;
  }

  if (t > earliest_time_) {
    Finalize(t);
    return;
  }

  // Earlier spike than previous earliest - reset candidates.
  earliest_time_ = t;
  candidates_.clear();
  candidates_.push_back(idx);
}

void FirstSpikeDecoder::Finalize(float /*t_now*/) {
  if (winner_.has_value() || candidates_.empty()) {
    return;
  }
  if (candidates_.size() == 1) {
    winner_ = candidates_.front();
    winner_time_ = earliest_time_;
  } else {
    std::uniform_int_distribution<int> dist(
        0, static_cast<int>(candidates_.size()) - 1);
    int pick = candidates_[dist(rng_)];
    winner_ = pick;
    winner_time_ = earliest_time_;
  }
  candidates_.clear();
}

std::optional<int> FirstSpikeDecoder::Result() {
  if (!winner_.has_value() && !candidates_.empty()) {
    Finalize(earliest_time_);
  }
  return winner_;
}

std::optional<int> FirstSpikeDecoder::ResultWithTimeout(float t_now) {
  if (winner_.has_value()) {
    return winner_;
  }
  if (!candidates_.empty() && (t_now - earliest_time_) >= 0.0F) {
    Finalize(t_now);
    return winner_;
  }
  if ((t_now - start_time_ms_) >= window_ms_) {
    window_expired_ = true;
    return winner_;  // likely empty
  }
  return std::nullopt;
}

}  // namespace senna::decoding

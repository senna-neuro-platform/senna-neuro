#include "core/plasticity/homeostasis.hpp"

#include <algorithm>

namespace senna::plasticity {

std::vector<float> Homeostasis::ComputeTheta(
    const std::vector<float>& theta_cur, const std::vector<float>& r_avg,
    float dt_ms, float global_activity) const {
  const int n = static_cast<int>(r_avg.size());
  if (n == 0 || static_cast<int>(theta_cur.size()) != n || dt_ms <= 0.0f) {
    return theta_cur;
  }
  const float dt_seconds = dt_ms * 1e-3f;

  std::vector<float> theta_new(theta_cur);

  for (int i = 0; i < n; ++i) {
    float mix = std::clamp(cfg_.global_mix, 0.0f, 1.0f);
    float activity = (global_activity >= 0.0f)
                         ? mix * global_activity + (1.0f - mix) * r_avg[i]
                         : r_avg[i];
    float freq_hz = activity / dt_seconds;
    float diff = freq_hz - cfg_.target_rate_hz;

    float theta = theta_cur[i];
    if (diff > 0.0f) {
      theta += cfg_.theta_step * diff;
    } else if (diff < 0.0f) {
      theta -= cfg_.theta_step * (-diff);
    }
    theta = std::clamp(theta, cfg_.theta_min, cfg_.theta_max);
    theta_new[i] = theta;
  }
  return theta_new;
}

}  // namespace senna::plasticity

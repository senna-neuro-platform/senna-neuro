#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/engine/network_builder.h"
#include "core/io/first_spike_decoder.h"
#include "core/metrics/metrics_collector.h"
#include "core/persistence/state_serializer.h"
#include "core/plasticity/supervisor.h"

namespace py = pybind11;

namespace {

constexpr std::size_t kMnistPixels = 28U * 28U;

struct BindingConfig {
    senna::core::engine::NetworkBuilderConfig network{};
    senna::core::domain::Time sample_duration_ms{50.0F};
    senna::core::domain::Weight supervision_spike_value{1.1F};
    std::uint32_t seed{42U};
};

template <typename T>
T yaml_or(const YAML::Node& node, const char* key, const T& fallback) {
    if (!node || !node[key]) {
        return fallback;
    }
    return node[key].as<T>();
}

template <typename T>
T yaml_required(const YAML::Node& node, const char* key, const char* section) {
    if (!node || !node[key]) {
        throw std::invalid_argument("Missing required config field: " + std::string(section) + "." +
                                    key);
    }
    return node[key].as<T>();
}

void validate_config(const BindingConfig& config) {
    const auto& lattice = config.network.lattice;
    if (lattice.width < 28U || lattice.height < 28U) {
        throw std::invalid_argument(
            "lattice.width and lattice.height must be >= 28 for MNIST mapping");
    }
    if (lattice.depth < 2U) {
        throw std::invalid_argument("lattice.depth must be >= 2");
    }
    if (lattice.processing_density < 0.0F || lattice.processing_density > 1.0F) {
        throw std::invalid_argument("lattice.processing_density must be in [0, 1]");
    }
    if (lattice.excitatory_ratio < 0.0F || lattice.excitatory_ratio > 1.0F) {
        throw std::invalid_argument("lattice.excitatory_ratio must be in [0, 1]");
    }
    if (lattice.output_neurons == 0U) {
        throw std::invalid_argument("lattice.output_neurons must be > 0");
    }
    if (lattice.neighbor_radius <= 0.0F) {
        throw std::invalid_argument("lattice.neighbor_radius must be > 0");
    }

    if (config.network.c_base <= 0.0F) {
        throw std::invalid_argument("synapse.c_base must be > 0");
    }
    if (config.network.min_weight < 0.0F || config.network.max_weight < 0.0F) {
        throw std::invalid_argument("synapse weights must be >= 0");
    }
    if (config.network.min_weight > config.network.max_weight) {
        throw std::invalid_argument("synapse.w_init_min must be <= synapse.w_init_max");
    }

    if (config.network.dt <= 0.0F) {
        throw std::invalid_argument("encoder.dt must be > 0");
    }
    if (config.sample_duration_ms <= 0.0F) {
        throw std::invalid_argument("encoder.duration_ms must be > 0");
    }
    if (config.network.input_spike_value <= 0.0F) {
        throw std::invalid_argument("encoder.spike_value must be > 0");
    }
    if (config.supervision_spike_value <= 0.0F) {
        throw std::invalid_argument("training.supervision_spike_value must be > 0");
    }
}

BindingConfig load_binding_config(const std::string& path) {
    BindingConfig config{};

    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("Config file does not exist: " + path);
    }

    const auto root = YAML::LoadFile(path);

    const auto neuron = root["neuron"];
    const auto stdp = root["stdp"];
    const auto homeostasis = root["homeostasis"];
    const auto structural = root["structural"];
    const auto decoder = root["decoder"];

    const auto lattice = root["lattice"];
    config.network.lattice.width = yaml_required<std::uint16_t>(lattice, "width", "lattice");
    config.network.lattice.height = yaml_required<std::uint16_t>(lattice, "height", "lattice");
    config.network.lattice.depth = yaml_required<std::uint16_t>(lattice, "depth", "lattice");
    config.network.lattice.processing_density =
        yaml_required<float>(lattice, "processing_density", "lattice");
    config.network.lattice.output_neurons =
        yaml_required<std::uint16_t>(lattice, "output_neurons", "lattice");
    config.network.lattice.excitatory_ratio =
        yaml_required<float>(lattice, "excitatory_ratio", "lattice");
    config.network.lattice.neighbor_radius =
        yaml_required<float>(lattice, "neighbor_radius", "lattice");

    const auto synapse = root["synapse"];
    config.network.c_base = yaml_required<senna::core::domain::Time>(synapse, "c_base", "synapse");

    if (synapse && synapse["w_init_range"] && synapse["w_init_range"].IsSequence() &&
        synapse["w_init_range"].size() >= 2U) {
        config.network.min_weight = synapse["w_init_range"][0U].as<senna::core::domain::Weight>();
        config.network.max_weight = synapse["w_init_range"][1U].as<senna::core::domain::Weight>();
    }

    config.network.min_weight =
        yaml_or<senna::core::domain::Weight>(synapse, "w_init_min", config.network.min_weight);
    config.network.max_weight =
        yaml_or<senna::core::domain::Weight>(synapse, "w_init_max", config.network.max_weight);

    const auto encoder = root["encoder"];
    config.network.input_spike_value =
        yaml_required<senna::core::domain::Weight>(encoder, "spike_value", "encoder");
    config.network.dt = yaml_required<senna::core::domain::Time>(encoder, "dt", "encoder");
    config.sample_duration_ms =
        yaml_required<senna::core::domain::Time>(encoder, "duration_ms", "encoder");

    const auto training = root["training"];
    config.seed = yaml_required<std::uint32_t>(training, "seed", "training");
    config.supervision_spike_value = yaml_or<senna::core::domain::Weight>(
        training, "supervision_spike_value", config.network.input_spike_value);

    const auto neuron_v_rest = yaml_required<float>(neuron, "V_rest", "neuron");
    const auto neuron_tau_m = yaml_required<float>(neuron, "tau_m", "neuron");
    const auto neuron_theta_base = yaml_required<float>(neuron, "theta_base", "neuron");
    const auto neuron_t_ref = yaml_required<float>(neuron, "t_ref", "neuron");
    const auto stdp_a_plus = yaml_required<float>(stdp, "A_plus", "stdp");
    const auto stdp_a_minus = yaml_required<float>(stdp, "A_minus", "stdp");
    const auto stdp_tau_plus = yaml_required<float>(stdp, "tau_plus", "stdp");
    const auto stdp_tau_minus = yaml_required<float>(stdp, "tau_minus", "stdp");
    const auto stdp_w_max = yaml_required<float>(stdp, "w_max", "stdp");
    const auto homeo_r_target = yaml_required<float>(homeostasis, "r_target", "homeostasis");
    const auto homeo_eta = yaml_required<float>(homeostasis, "eta_homeo", "homeostasis");
    const auto homeo_theta_min = yaml_required<float>(homeostasis, "theta_min", "homeostasis");
    const auto homeo_theta_max = yaml_required<float>(homeostasis, "theta_max", "homeostasis");
    const auto structural_w_min = yaml_required<float>(structural, "w_min", "structural");
    const auto structural_n_prune =
        yaml_required<std::uint32_t>(structural, "N_prune", "structural");
    const auto encoder_max_rate = yaml_required<float>(encoder, "max_rate", "encoder");
    const auto decoder_w_wta = yaml_required<float>(decoder, "W_wta", "decoder");
    const auto training_epochs = yaml_required<std::uint32_t>(training, "epochs", "training");

    if (neuron_tau_m <= 0.0F || neuron_theta_base <= 0.0F || neuron_t_ref < 0.0F ||
        !std::isfinite(neuron_v_rest)) {
        throw std::invalid_argument("neuron parameters are out of valid range");
    }
    if (stdp_a_plus <= 0.0F || stdp_a_minus <= 0.0F || stdp_tau_plus <= 0.0F ||
        stdp_tau_minus <= 0.0F) {
        throw std::invalid_argument("stdp parameters must be > 0");
    }
    if (stdp_w_max <= 0.0F) {
        throw std::invalid_argument("stdp.w_max must be > 0");
    }
    if (homeo_r_target < 0.0F || homeo_eta <= 0.0F) {
        throw std::invalid_argument("homeostasis parameters are out of valid range");
    }
    if (homeo_theta_min > homeo_theta_max) {
        throw std::invalid_argument("homeostasis.theta_min must be <= homeostasis.theta_max");
    }
    if (structural_w_min < 0.0F || structural_n_prune == 0U) {
        throw std::invalid_argument("structural.N_prune must be > 0");
    }
    if (encoder_max_rate <= 0.0F) {
        throw std::invalid_argument("encoder.max_rate must be > 0");
    }
    if (decoder_w_wta >= 0.0F) {
        throw std::invalid_argument("decoder.W_wta must be negative");
    }
    if (training_epochs == 0U) {
        throw std::invalid_argument("training.epochs must be > 0");
    }

    validate_config(config);
    return config;
}

std::vector<senna::core::domain::NeuronId> build_sensor_map(
    const senna::core::domain::Lattice& lattice) {
    std::vector<senna::core::domain::NeuronId> sensor{};
    sensor.reserve(kMnistPixels);

    for (std::uint16_t y = 0U; y < 28U; ++y) {
        for (std::uint16_t x = 0U; x < 28U; ++x) {
            const auto id = lattice.neuron_id_at(senna::core::domain::Coord3D{x, y, 0U});
            if (!id.has_value()) {
                throw std::runtime_error("Sensor layer is incomplete for MNIST 28x28 mapping");
            }
            sensor.push_back(*id);
        }
    }

    return sensor;
}

std::vector<senna::core::domain::NeuronId> find_output_neurons(
    const std::vector<senna::core::domain::Neuron>& neurons, const std::uint16_t output_layer_z) {
    std::vector<senna::core::domain::NeuronId> output{};
    for (const auto& neuron : neurons) {
        if (neuron.position().z == output_layer_z) {
            output.push_back(neuron.id());
        }
    }

    if (output.empty()) {
        throw std::runtime_error("Output layer is empty");
    }

    std::sort(output.begin(), output.end(),
              [](const auto lhs, const auto rhs) { return lhs < rhs; });
    return output;
}

class PyNetworkHandle final {
   public:
    explicit PyNetworkHandle(const std::string& config_path = "configs/default.yaml")
        : config_path_(config_path),
          config_(load_binding_config(config_path_)),
          network_(senna::core::engine::NetworkBuilder{config_.network}.build_ptr(config_.seed)),
          metrics_collector_(network_->neurons(), network_->synapse_count()),
          sensor_map_(build_sensor_map(network_->lattice())),
          output_neurons_(
              find_output_neurons(network_->neurons(), network_->lattice().config().depth - 1U)),
          output_set_(output_neurons_.begin(), output_neurons_.end()),
          decoder_(output_neurons_),
          supervisor_({config_.supervision_spike_value}) {
        attach_observers();
        metrics_collector_.set_synapse_count(network_->synapse_count());
        rng_state_ = seed_to_state(config_.seed);
    }

    void load_sample(const std::vector<std::uint8_t>& image, const int label,
                     const bool is_train = true) {
        if (image.size() != kMnistPixels) {
            throw std::invalid_argument("load_sample expects flattened 28x28 image (784 values)");
        }
        if (label < 0 || label > 9) {
            throw std::invalid_argument("label must be in [0, 9]");
        }

        sample_image_ = image;
        sample_label_ = label;
        sample_is_train_ = is_train && !eval_mode_;
        sample_loaded_ = true;

        inject_encoded_sample();
    }

    void step(const std::size_t n_ticks) {
        if (n_ticks == 0U) {
            return;
        }

        for (std::size_t i = 0U; i < n_ticks; ++i) {
            static_cast<void>(network_->tick());
        }

        if (!sample_loaded_) {
            return;
        }

        finalize_prediction();
        sample_loaded_ = false;
    }

    [[nodiscard]] int get_prediction() const noexcept { return last_prediction_; }

    [[nodiscard]] py::dict get_metrics() {
        refresh_accuracy_metrics();
        auto map = metrics_collector_.as_metric_map();

        py::dict out{};
        for (const auto& [key, value] : map) {
            out[py::str(key)] = py::float_(value);
        }

        out[py::str("prediction")] = py::int_(last_prediction_);
        out[py::str("label")] = py::int_(sample_label_.value_or(-1));
        out[py::str("train_seen")] = py::int_(train_seen_);
        out[py::str("train_correct")] = py::int_(train_correct_);
        out[py::str("test_seen")] = py::int_(test_seen_);
        out[py::str("test_correct")] = py::int_(test_correct_);
        return out;
    }

    void save_state(const std::string& path) const {
        const auto state = senna::core::persistence::StateSerializer::capture(
            network_->neurons(), network_->synapses(), network_->event_queue(),
            network_->time_manager(), rng_state_);
        senna::core::persistence::StateSerializer::save_state(path, state);

        std::ofstream meta(path + ".meta", std::ios::trunc);
        meta << config_path_ << "\n";
        meta << last_prediction_ << "\n";
        meta << train_seen_ << " " << train_correct_ << "\n";
        meta << test_seen_ << " " << test_correct_ << "\n";
    }

    static std::shared_ptr<PyNetworkHandle> load_state(const std::string& path,
                                                       const std::string& config_path = "") {
        std::string chosen_config = config_path;

        std::ifstream meta(path + ".meta");
        if (!meta.fail()) {
            std::string meta_config{};
            if (std::getline(meta, meta_config) && chosen_config.empty() && !meta_config.empty()) {
                chosen_config = meta_config;
            }
        }

        if (chosen_config.empty()) {
            chosen_config = "configs/default.yaml";
        }

        auto handle = std::make_shared<PyNetworkHandle>(chosen_config);

        const auto state = senna::core::persistence::StateSerializer::load_state(path);
        senna::core::persistence::StateSerializer::restore(state, handle->network_->neurons(),
                                                           handle->network_->synapses(),
                                                           handle->network_->event_queue());
        handle->network_->time_manager().reset(state.elapsed);
        handle->rng_state_ = state.rng_state;

        std::ifstream meta_reload(path + ".meta");
        if (!meta_reload.fail()) {
            std::string ignored{};
            static_cast<void>(std::getline(meta_reload, ignored));
            meta_reload >> handle->last_prediction_;
            meta_reload >> handle->train_seen_ >> handle->train_correct_;
            meta_reload >> handle->test_seen_ >> handle->test_correct_;
        }

        handle->refresh_accuracy_metrics();
        handle->metrics_collector_.set_synapse_count(handle->network_->synapse_count());
        return handle;
    }

    void inject_noise(const double sigma) {
        if (sigma < 0.0) {
            throw std::invalid_argument("sigma must be non-negative");
        }

        for (auto& neuron : network_->neurons()) {
            const auto noise = static_cast<float>((next_unit() - 0.5) * 2.0 * sigma);
            neuron.set_threshold(std::max(0.05F, neuron.threshold() + noise));
        }
    }

    void remove_neurons(const double fraction) {
        if (fraction < 0.0 || fraction > 1.0) {
            throw std::invalid_argument("fraction must be in [0, 1]");
        }

        auto& neurons = network_->neurons();
        const auto remove_count = static_cast<std::size_t>(std::floor(fraction * neurons.size()));
        if (remove_count == 0U) {
            return;
        }

        for (std::size_t i = 0U; i < remove_count; ++i) {
            const auto index =
                static_cast<std::size_t>(next_unit() * neurons.size()) % neurons.size();
            neurons[index].set_threshold(1e9F);
            neurons[index].set_average_rate(0.0F);
        }
    }

    void set_eval_mode(const bool enabled) noexcept { eval_mode_ = enabled; }

    void supervise(const int expected_label) {
        if (expected_label < 0 || expected_label > 9) {
            throw std::invalid_argument("expected_label must be in [0, 9]");
        }

        const auto was_correct = (last_prediction_ == expected_label);
        const auto correction = supervisor_.correction_event(last_prediction_, expected_label,
                                                             output_neurons_, network_->elapsed());
        if (!correction.has_value()) {
            return;
        }

        output_spikes_.clear();
        network_->inject_event(*correction);
        static_cast<void>(network_->tick());
        const auto corrected = decoder_.decode(output_spikes_);
        last_prediction_ = corrected >= 0 ? corrected : expected_label;
        if (sample_is_train_ && sample_label_.has_value() && *sample_label_ == expected_label &&
            !was_correct && last_prediction_ == expected_label) {
            ++train_correct_;
            refresh_accuracy_metrics();
        }
        output_spikes_.clear();
        metrics_collector_.set_synapse_count(network_->synapse_count());
    }

   private:
    void attach_observers() {
        network_->add_spike_observer([this](const senna::core::domain::SpikeEvent& spike) {
            metrics_collector_.on_spike(spike);
            if (output_set_.contains(spike.source)) {
                output_spikes_.push_back(spike);
            }
        });

        network_->add_tick_observer(
            [this](const senna::core::domain::Time t_start, const senna::core::domain::Time t_end) {
                metrics_collector_.on_tick(t_start, t_end);
            });
    }

    void inject_encoded_sample() {
        if (!sample_loaded_) {
            return;
        }

        const auto base_time = network_->elapsed();
        const auto dt = std::max(0.01F, config_.network.dt);
        const auto sample_window = std::max(dt, config_.sample_duration_ms);
        constexpr std::size_t kMaxSpikesPerPixel = 3U;
        const auto pulse_stride =
            sample_window / static_cast<senna::core::domain::Time>(kMaxSpikesPerPixel);

        for (std::size_t index = 0U; index < sample_image_.size(); ++index) {
            const auto pixel = sample_image_[index];
            if (pixel == 0U) {
                continue;
            }

            const auto normalized = static_cast<float>(pixel) / 255.0F;
            if (normalized < 0.08F) {
                continue;
            }

            const auto pulse_count = std::clamp<std::size_t>(
                static_cast<std::size_t>(std::ceil(normalized * kMaxSpikesPerPixel)), 1U,
                kMaxSpikesPerPixel);
            const auto value = static_cast<senna::core::domain::Weight>(
                config_.network.input_spike_value * normalized);

            for (std::size_t pulse = 0U; pulse < pulse_count; ++pulse) {
                const auto deterministic_jitter =
                    static_cast<senna::core::domain::Time>((index % 17U) * dt * 0.01F);
                const auto local_offset = static_cast<senna::core::domain::Time>(
                    static_cast<senna::core::domain::Time>(pulse) * pulse_stride +
                    deterministic_jitter);
                if (local_offset > sample_window) {
                    continue;
                }

                network_->inject_event(senna::core::domain::SpikeEvent{
                    sensor_map_[index],
                    sensor_map_[index],
                    base_time + local_offset,
                    value,
                });
            }
        }

        output_spikes_.clear();
    }

    void finalize_prediction() {
        last_prediction_ = decoder_.decode(output_spikes_);

        if (sample_is_train_) {
            ++train_seen_;
            if (sample_label_.has_value() && last_prediction_ == *sample_label_) {
                ++train_correct_;
            }
        } else {
            ++test_seen_;
            if (sample_label_.has_value() && last_prediction_ == *sample_label_) {
                ++test_correct_;
            }
        }

        refresh_accuracy_metrics();
        metrics_collector_.set_synapse_count(network_->synapse_count());
        output_spikes_.clear();
    }

    void refresh_accuracy_metrics() {
        const auto train_acc = train_seen_ == 0U ? 0.0
                                                 : static_cast<double>(train_correct_) /
                                                       static_cast<double>(train_seen_);
        const auto test_acc =
            test_seen_ == 0U ? 0.0
                             : static_cast<double>(test_correct_) / static_cast<double>(test_seen_);
        metrics_collector_.set_train_accuracy(train_acc);
        metrics_collector_.set_test_accuracy(test_acc);
    }

    [[nodiscard]] static std::uint64_t seed_to_state(const std::uint32_t seed) noexcept {
        auto state =
            static_cast<std::uint64_t>(seed) ^ static_cast<std::uint64_t>(0x9e3779b97f4a7c15ULL);
        if (state == 0U) {
            state = 1U;
        }
        return state;
    }

    double next_unit() noexcept {
        rng_state_ = (rng_state_ * 6364136223846793005ULL) + 1ULL;
        constexpr double kDenominator = static_cast<double>(1ULL << 53U);
        return static_cast<double>((rng_state_ >> 11U) & ((1ULL << 53U) - 1ULL)) / kDenominator;
    }

    std::string config_path_{};
    BindingConfig config_{};
    std::unique_ptr<senna::core::engine::Network> network_{};

    senna::core::metrics::MetricsCollector metrics_collector_;
    std::vector<senna::core::domain::NeuronId> sensor_map_{};
    std::vector<senna::core::domain::NeuronId> output_neurons_{};
    std::unordered_set<senna::core::domain::NeuronId> output_set_{};
    senna::core::io::FirstSpikeDecoder decoder_;
    senna::core::plasticity::Supervisor supervisor_;

    std::vector<std::uint8_t> sample_image_ = std::vector<std::uint8_t>(kMnistPixels, 0U);
    std::optional<int> sample_label_{};
    bool sample_loaded_{false};
    bool sample_is_train_{true};
    bool eval_mode_{false};

    std::vector<senna::core::domain::SpikeEvent> output_spikes_{};
    int last_prediction_{-1};

    std::uint64_t rng_state_{0x123456789abcdef0ULL};

    std::size_t train_seen_{0U};
    std::size_t train_correct_{0U};
    std::size_t test_seen_{0U};
    std::size_t test_correct_{0U};
};

}  // namespace

PYBIND11_MODULE(senna_core, module) {
    module.doc() = "SENNA Neuro Python bindings";

    py::class_<PyNetworkHandle, std::shared_ptr<PyNetworkHandle>>(module, "NetworkHandle")
        .def(py::init<const std::string&>(), py::arg("config_path") = "configs/default.yaml")
        .def("load_sample", &PyNetworkHandle::load_sample, py::arg("image"), py::arg("label"),
             py::arg("is_train") = true)
        .def("step", &PyNetworkHandle::step, py::arg("n_ticks"))
        .def("get_prediction", &PyNetworkHandle::get_prediction)
        .def("get_metrics", &PyNetworkHandle::get_metrics)
        .def("save_state", &PyNetworkHandle::save_state, py::arg("path"))
        .def("inject_noise", &PyNetworkHandle::inject_noise, py::arg("sigma"))
        .def("remove_neurons", &PyNetworkHandle::remove_neurons, py::arg("fraction"))
        .def("supervise", &PyNetworkHandle::supervise, py::arg("expected_label"))
        .def("set_eval_mode", &PyNetworkHandle::set_eval_mode, py::arg("enabled"));

    module.def(
        "create_network",
        [](const std::string& config_path) {
            return std::make_shared<PyNetworkHandle>(config_path);
        },
        py::arg("config_path") = "configs/default.yaml");

    module.def(
        "load_sample",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const std::vector<std::uint8_t>& image,
           const int label, const bool is_train) { handle->load_sample(image, label, is_train); },
        py::arg("handle"), py::arg("image"), py::arg("label"), py::arg("is_train") = true);

    module.def(
        "step",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const std::size_t n_ticks) {
            handle->step(n_ticks);
        },
        py::arg("handle"), py::arg("n_ticks"));

    module.def(
        "get_prediction",
        [](const std::shared_ptr<PyNetworkHandle>& handle) { return handle->get_prediction(); },
        py::arg("handle"));

    module.def(
        "get_metrics",
        [](const std::shared_ptr<PyNetworkHandle>& handle) { return handle->get_metrics(); },
        py::arg("handle"));

    module.def(
        "save_state",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const std::string& path) {
            handle->save_state(path);
        },
        py::arg("handle"), py::arg("path"));

    module.def("load_state", &PyNetworkHandle::load_state, py::arg("path"),
               py::arg("config_path") = "");

    module.def(
        "load_state",
        [](const std::shared_ptr<PyNetworkHandle>& /*handle*/, const std::string& path,
           const std::string& config_path) {
            return PyNetworkHandle::load_state(path, config_path);
        },
        py::arg("handle"), py::arg("path"), py::arg("config_path") = "");

    module.def(
        "inject_noise",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const double sigma) {
            handle->inject_noise(sigma);
        },
        py::arg("handle"), py::arg("sigma"));

    module.def(
        "remove_neurons",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const double fraction) {
            handle->remove_neurons(fraction);
        },
        py::arg("handle"), py::arg("fraction"));

    module.def(
        "supervise",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const int expected_label) {
            handle->supervise(expected_label);
        },
        py::arg("handle"), py::arg("expected_label"));
}

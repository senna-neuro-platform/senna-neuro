#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

constexpr std::size_t kMnistPixels = static_cast<std::size_t>(28U) * static_cast<std::size_t>(28U);

struct BindingConfig {
    senna::core::engine::NetworkBuilderConfig network{};
    senna::core::domain::Time sample_duration_ms{50.0F};
    senna::core::domain::Weight supervision_spike_value{1.1F};
    float max_rate_hz{120.0F};
    senna::core::domain::Weight max_synapse_weight{1.5F};
    senna::core::domain::Weight decoder_wta_weight{6.0F};
    senna::core::domain::Weight supervision_learning_rate{0.02F};
    std::uint32_t seed{42U};
};

struct BatchExecutionStats {
    std::size_t completed{0U};
    std::size_t correct{0U};
};

using ImageArray = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;

struct BatchArrayView {
    const std::uint8_t* images{nullptr};
    const std::int32_t* labels{nullptr};
    std::size_t sample_count{0U};
};

struct EncodedInputPulse {
    senna::core::domain::NeuronId sensor_id{};
    senna::core::domain::Time local_offset{0.0F};
    senna::core::domain::Weight value{0.0F};
};

using EncodedSamplePlan = std::vector<EncodedInputPulse>;

struct SampleCacheKey {
    std::array<std::uint8_t, kMnistPixels> pixels{};

    [[nodiscard]] bool operator==(const SampleCacheKey& other) const noexcept = default;
};

struct SampleCacheKeyHash {
    [[nodiscard]] std::size_t operator()(const SampleCacheKey& key) const noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (const auto pixel : key.pixels) {
            hash ^= static_cast<std::uint64_t>(pixel);
            hash *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(hash);
    }
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

[[nodiscard]] const char* neuron_type_name(const senna::core::domain::NeuronType type) noexcept {
    switch (type) {
        case senna::core::domain::NeuronType::Excitatory:
            return "excitatory";
        case senna::core::domain::NeuronType::Inhibitory:
            return "inhibitory";
    }
    return "unknown";
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
    if (config.max_synapse_weight < config.network.max_weight) {
        throw std::invalid_argument("stdp.w_max must be >= synapse.w_init_max");
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
    if (config.max_rate_hz <= 0.0F) {
        throw std::invalid_argument("encoder.max_rate must be > 0");
    }
    if (config.decoder_wta_weight <= 0.0F) {
        throw std::invalid_argument("decoder.W_wta absolute value must be > 0");
    }
    if (config.supervision_learning_rate <= 0.0F) {
        throw std::invalid_argument("training.learning_rate must be > 0");
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
    config.supervision_learning_rate = yaml_or<senna::core::domain::Weight>(
        training, "learning_rate", config.supervision_learning_rate);
    const auto training_target_accuracy = yaml_or<float>(training, "target_accuracy", 0.85F);

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

    config.max_synapse_weight = stdp_w_max;
    config.max_rate_hz = encoder_max_rate;
    config.decoder_wta_weight = std::abs(decoder_w_wta);

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
    if (training_target_accuracy < 0.0F || training_target_accuracy > 1.0F) {
        throw std::invalid_argument("training.target_accuracy must be in [0, 1]");
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

BatchArrayView validate_batch_arrays(const ImageArray& images, const LabelArray& labels) {
    const auto image_info = images.request();
    const auto label_info = labels.request();

    if (image_info.ndim != 2 || image_info.shape[1] != static_cast<py::ssize_t>(kMnistPixels)) {
        throw std::invalid_argument("batch arrays expect images shaped as [N, 784]");
    }
    if (label_info.ndim != 1) {
        throw std::invalid_argument("batch arrays expect labels shaped as [N]");
    }

    const auto sample_count = static_cast<std::size_t>(image_info.shape[0]);
    if (sample_count != static_cast<std::size_t>(label_info.shape[0])) {
        throw std::invalid_argument("batch images and labels must have the same size");
    }

    return BatchArrayView{
        static_cast<const std::uint8_t*>(image_info.ptr),
        static_cast<const std::int32_t*>(label_info.ptr),
        sample_count,
    };
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
          decoder_(output_neurons_, config_.decoder_wta_weight),
          supervisor_({config_.supervision_spike_value}) {
        attach_observers();
        metrics_collector_.set_synapse_count(network_->synapse_count());
        rng_state_ = seed_to_state(config_.seed);
        sample_spike_counts_.assign(network_->neurons().size(), 0U);
        sample_plan_cache_.reserve(4096U);
    }

    void load_sample(const std::vector<std::uint8_t>& image, const int label,
                     const bool is_train = true) {
        load_sample_view(std::span<const std::uint8_t>(image.data(), image.size()), label,
                         is_train);
    }

    [[nodiscard]] BatchExecutionStats batch_train(
        const std::vector<std::vector<std::uint8_t>>& images, const std::vector<int>& labels,
        const std::size_t ticks_per_sample) {
        set_eval_mode(false);
        return run_batch(images, labels, ticks_per_sample, true);
    }

    [[nodiscard]] BatchExecutionStats batch_train_array(const BatchArrayView& batch,
                                                        const std::size_t ticks_per_sample) {
        set_eval_mode(false);
        return run_batch(batch, ticks_per_sample, true);
    }

    [[nodiscard]] BatchExecutionStats batch_evaluate(
        const std::vector<std::vector<std::uint8_t>>& images, const std::vector<int>& labels,
        const std::size_t ticks_per_sample) {
        set_eval_mode(true);
        return run_batch(images, labels, ticks_per_sample, false);
    }

    [[nodiscard]] BatchExecutionStats batch_evaluate_array(const BatchArrayView& batch,
                                                           const std::size_t ticks_per_sample) {
        set_eval_mode(true);
        return run_batch(batch, ticks_per_sample, false);
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

    [[nodiscard]] py::dict get_lattice() const {
        const auto& lattice = network_->lattice();
        const auto& config = lattice.config();
        const auto& neurons = lattice.neurons();

        py::list neuron_items{};
        for (const auto& neuron : neurons) {
            py::dict item{};
            item[py::str("id")] = py::int_(neuron.id());
            item[py::str("x")] = py::int_(neuron.position().x);
            item[py::str("y")] = py::int_(neuron.position().y);
            item[py::str("z")] = py::int_(neuron.position().z);
            item[py::str("type")] =
                py::str(neuron.type() == senna::core::domain::NeuronType::Excitatory
                            ? (neuron.position().z == config.depth - 1U ? "output" : "excitatory")
                            : neuron_type_name(neuron.type()));
            neuron_items.append(std::move(item));
        }

        py::dict payload{};
        payload[py::str("width")] = py::int_(config.width);
        payload[py::str("height")] = py::int_(config.height);
        payload[py::str("depth")] = py::int_(config.depth);
        payload[py::str("neuronCount")] = py::int_(neurons.size());
        payload[py::str("neurons")] = std::move(neuron_items);
        return payload;
    }

    [[nodiscard]] py::list step_with_trace(const std::size_t n_ticks) {
        py::list frames{};
        if (n_ticks == 0U) {
            return frames;
        }

        for (std::size_t i = 0U; i < n_ticks; ++i) {
            static_cast<void>(network_->tick());
            frames.append(build_tick_frame(i + 1U));
        }

        if (!sample_loaded_) {
            return frames;
        }

        finalize_prediction();
        sample_loaded_ = false;
        return frames;
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
                                                       const char* config_path = nullptr) {
        std::string chosen_config = config_path == nullptr ? "" : config_path;

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
        const auto neuron_count = neurons.size();
        const auto remove_count =
            static_cast<std::size_t>(std::floor(fraction * static_cast<double>(neuron_count)));
        if (remove_count == 0U) {
            return;
        }

        for (std::size_t i = 0U; i < remove_count; ++i) {
            const auto random_index =
                static_cast<std::size_t>(next_unit() * static_cast<double>(neuron_count));
            const auto index = random_index % neuron_count;
            neurons[index].set_threshold(1e9F);
            neurons[index].set_average_rate(0.0F);
        }
    }

    void set_eval_mode(const bool enabled) noexcept { eval_mode_ = enabled; }

    void supervise(const int expected_label) {
        if (expected_label < 0 || expected_label > 9) {
            throw std::invalid_argument("expected_label must be in [0, 9]");
        }

        const auto predicted_before = last_prediction_;
        const auto was_correct = (last_prediction_ == expected_label);
        if (sample_is_train_) {
            apply_supervised_weight_update(expected_label, predicted_before);
        }

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
        emit_wta_for_prediction(last_prediction_);
        if (sample_is_train_ && sample_label_.has_value() && *sample_label_ == expected_label &&
            !was_correct && last_prediction_ == expected_label) {
            ++train_correct_;
            refresh_accuracy_metrics();
        }
        output_spikes_.clear();
        metrics_collector_.set_synapse_count(network_->synapse_count());
    }

   private:
    void load_sample_view(const std::span<const std::uint8_t> image, const int label,
                          const bool is_train = true) {
        if (image.size() != kMnistPixels) {
            throw std::invalid_argument("load_sample expects flattened 28x28 image (784 values)");
        }
        if (label < 0 || label > 9) {
            throw std::invalid_argument("label must be in [0, 9]");
        }

        current_sample_plan_ = get_or_build_sample_plan(image);
        sample_label_ = label;
        sample_is_train_ = is_train && !eval_mode_;
        sample_loaded_ = true;

        network_->reset_between_samples();
        reset_sample_activity();
        inject_encoded_sample();
    }

    [[nodiscard]] bool is_active_pixel(const std::uint8_t pixel) const noexcept {
        return (static_cast<float>(pixel) / 255.0F) >= 0.08F;
    }

    [[nodiscard]] bool is_valid_output_label(const int label) const noexcept {
        return label >= 0 && static_cast<std::size_t>(label) < output_neurons_.size();
    }

    [[nodiscard]] senna::core::domain::NeuronId output_neuron_for_label(const int label) const {
        if (!is_valid_output_label(label)) {
            throw std::out_of_range("Output label is out of range");
        }
        return output_neurons_[static_cast<std::size_t>(label)];
    }

    void apply_supervised_weight_update(const int expected_label, const int predicted_label) {
        if (!is_valid_output_label(expected_label)) {
            return;
        }

        const auto expected_output = output_neuron_for_label(expected_label);
        const auto has_predicted =
            is_valid_output_label(predicted_label) && predicted_label != expected_label;
        if (!has_predicted && last_prediction_ == expected_label) {
            return;
        }

        static_cast<void>(expected_output);
        static_cast<void>(supervisor_.apply_output_weight_update(
            predicted_label, expected_label, output_neurons_, sample_spike_counts_,
            network_->synapses(), std::max(0.0001F, config_.supervision_learning_rate),
            config_.network.min_weight, config_.max_synapse_weight));
    }

    void emit_wta_for_prediction(const int prediction) {
        if (!is_valid_output_label(prediction)) {
            return;
        }

        const auto winner = output_neuron_for_label(prediction);
        const auto inhibitory = decoder_.winner_take_all_events(winner, network_->elapsed());
        for (const auto& event : inhibitory) {
            network_->inject_event(event);
        }
    }

    void attach_observers() {
        network_->add_spike_observer([this](const senna::core::domain::SpikeEvent& spike) {
            metrics_collector_.on_spike(spike);
            record_sample_activity(spike);
            if (output_set_.contains(spike.source)) {
                output_spikes_.push_back(spike);
                // Real-time WTA: first output spike inhibits all other output neurons
                const auto inhibitory =
                    decoder_.winner_take_all_events(spike.source, spike.arrival);
                for (const auto& event : inhibitory) {
                    network_->inject_event(event);
                }
            }
        });

        network_->add_tick_observer(
            [this](const senna::core::domain::Time t_start, const senna::core::domain::Time t_end) {
                metrics_collector_.on_tick(t_start, t_end);
            });
    }

    [[nodiscard]] EncodedSamplePlan build_sample_plan(
        const std::span<const std::uint8_t> image) const {
        EncodedSamplePlan plan{};

        const auto dt = std::max(0.01F, config_.network.dt);
        const auto sample_window = std::max(dt, config_.sample_duration_ms);
        const auto max_spikes_per_pixel = std::clamp<std::size_t>(
            static_cast<std::size_t>(std::round(
                (static_cast<double>(config_.max_rate_hz) * static_cast<double>(sample_window)) /
                1000.0)),
            1U, 20U);
        const auto pulse_stride =
            sample_window / static_cast<senna::core::domain::Time>(max_spikes_per_pixel);

        plan.reserve(kMnistPixels / 4U);
        for (std::size_t index = 0U; index < image.size(); ++index) {
            const auto pixel = image[index];
            if (pixel == 0U || !is_active_pixel(pixel)) {
                continue;
            }

            const auto normalized = static_cast<float>(pixel) / 255.0F;
            const auto pulse_count = std::clamp<std::size_t>(
                static_cast<std::size_t>(
                    std::ceil(normalized * static_cast<float>(max_spikes_per_pixel))),
                1U, max_spikes_per_pixel);
            const auto value = static_cast<senna::core::domain::Weight>(
                config_.network.input_spike_value * normalized);
            const auto sensor_id = sensor_map_[index];
            const auto deterministic_jitter =
                static_cast<senna::core::domain::Time>(index % 17U) * dt * 0.01F;

            for (std::size_t pulse = 0U; pulse < pulse_count; ++pulse) {
                const auto local_offset = static_cast<senna::core::domain::Time>(
                    static_cast<senna::core::domain::Time>(pulse) * pulse_stride +
                    deterministic_jitter);
                if (local_offset > sample_window) {
                    continue;
                }

                plan.push_back(EncodedInputPulse{sensor_id, local_offset, value});
            }
        }

        return plan;
    }

    [[nodiscard]] std::shared_ptr<const EncodedSamplePlan> get_or_build_sample_plan(
        const std::span<const std::uint8_t> image) {
        SampleCacheKey key{};
        std::copy(image.begin(), image.end(), key.pixels.begin());

        if (const auto found = sample_plan_cache_.find(key); found != sample_plan_cache_.end()) {
            return found->second;
        }

        auto plan = std::make_shared<const EncodedSamplePlan>(build_sample_plan(image));
        const auto insert_result = sample_plan_cache_.emplace(key, std::move(plan));
        return insert_result.first->second;
    }

    void inject_encoded_sample() {
        if (!sample_loaded_ || !current_sample_plan_) {
            return;
        }

        const auto base_time = network_->elapsed();
        for (const auto& pulse : *current_sample_plan_) {
            network_->inject_event(senna::core::domain::SpikeEvent{
                pulse.sensor_id,
                pulse.sensor_id,
                base_time + pulse.local_offset,
                pulse.value,
            });
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

    [[nodiscard]] BatchExecutionStats run_batch(
        const std::vector<std::vector<std::uint8_t>>& images, const std::vector<int>& labels,
        const std::size_t ticks_per_sample, const bool is_train) {
        if (images.size() != labels.size()) {
            throw std::invalid_argument("batch images and labels must have the same size");
        }

        BatchExecutionStats stats{};
        for (std::size_t index = 0U; index < images.size(); ++index) {
            load_sample(images[index], labels[index], is_train);
            step(ticks_per_sample);

            auto prediction = get_prediction();
            if (is_train && prediction != labels[index]) {
                supervise(labels[index]);
                prediction = get_prediction();
            }

            ++stats.completed;
            if (prediction == labels[index]) {
                ++stats.correct;
            }
        }

        return stats;
    }

    [[nodiscard]] BatchExecutionStats run_batch(const BatchArrayView& batch,
                                                const std::size_t ticks_per_sample,
                                                const bool is_train) {
        BatchExecutionStats stats{};
        for (std::size_t index = 0U; index < batch.sample_count; ++index) {
            const auto image =
                std::span<const std::uint8_t>(batch.images + (index * kMnistPixels), kMnistPixels);
            const auto label = static_cast<int>(batch.labels[index]);

            load_sample_view(image, label, is_train);
            step(ticks_per_sample);

            auto prediction = get_prediction();
            if (is_train && prediction != label) {
                supervise(label);
                prediction = get_prediction();
            }

            ++stats.completed;
            if (prediction == label) {
                ++stats.correct;
            }
        }

        return stats;
    }

    void reset_sample_activity() {
        for (const auto neuron_id : active_sample_neurons_) {
            const auto index = static_cast<std::size_t>(neuron_id);
            if (index < sample_spike_counts_.size()) {
                sample_spike_counts_[index] = 0U;
            }
        }
        active_sample_neurons_.clear();
    }

    void record_sample_activity(const senna::core::domain::SpikeEvent& spike) {
        const auto index = static_cast<std::size_t>(spike.source);
        if (index >= sample_spike_counts_.size()) {
            return;
        }

        if (sample_spike_counts_[index] == 0U) {
            active_sample_neurons_.push_back(spike.source);
        }
        if (sample_spike_counts_[index] < std::numeric_limits<std::uint16_t>::max()) {
            ++sample_spike_counts_[index];
        }
    }

    [[nodiscard]] py::dict build_tick_frame(const std::size_t tick_index) const {
        const auto& neurons = network_->neurons();
        const auto& spikes = network_->emitted_spikes_last_tick();

        std::unordered_set<senna::core::domain::NeuronId> active_ids{};
        py::list active_neurons{};
        for (const auto& spike : spikes) {
            if (!active_ids.insert(spike.source).second) {
                continue;
            }

            const auto neuron_index = static_cast<std::size_t>(spike.source);
            if (neuron_index >= neurons.size()) {
                continue;
            }

            const auto& neuron = neurons.at(neuron_index);
            py::dict item{};
            item[py::str("id")] = py::int_(neuron.id());
            item[py::str("x")] = py::int_(neuron.position().x);
            item[py::str("y")] = py::int_(neuron.position().y);
            item[py::str("z")] = py::int_(neuron.position().z);
            item[py::str("type")] =
                py::str(neuron.type() == senna::core::domain::NeuronType::Excitatory &&
                                neuron.position().z == network_->lattice().config().depth - 1U
                            ? "output"
                            : neuron_type_name(neuron.type()));
            item[py::str("fired")] = py::bool_(true);
            active_neurons.append(std::move(item));
        }

        py::dict frame{};
        frame[py::str("tick")] = py::int_(tick_index);
        frame[py::str("neurons")] = std::move(active_neurons);
        frame[py::str("activeCount")] = py::int_(active_ids.size());
        frame[py::str("totalNeurons")] = py::int_(neurons.size());
        return frame;
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

    std::shared_ptr<const EncodedSamplePlan> current_sample_plan_{};
    std::unordered_map<SampleCacheKey, std::shared_ptr<const EncodedSamplePlan>, SampleCacheKeyHash>
        sample_plan_cache_{};
    std::optional<int> sample_label_{};
    bool sample_loaded_{false};
    bool sample_is_train_{true};
    bool eval_mode_{false};

    std::vector<senna::core::domain::SpikeEvent> output_spikes_{};
    std::vector<std::uint16_t> sample_spike_counts_{};
    std::vector<senna::core::domain::NeuronId> active_sample_neurons_{};
    int last_prediction_{-1};

    std::uint64_t rng_state_{0x123456789abcdef0ULL};

    std::size_t train_seen_{0U};
    std::size_t train_correct_{0U};
    std::size_t test_seen_{0U};
    std::size_t test_correct_{0U};
};

[[nodiscard]] py::dict batch_result_dict(PyNetworkHandle& handle,
                                         const BatchExecutionStats& stats) {
    auto result = handle.get_metrics();
    result[py::str("completed")] = py::int_(stats.completed);
    result[py::str("correct")] = py::int_(stats.correct);
    result[py::str("batch_accuracy")] =
        py::float_(stats.completed == 0U
                       ? 0.0
                       : static_cast<double>(stats.correct) / static_cast<double>(stats.completed));
    return result;
}

}  // namespace

PYBIND11_MODULE(senna_core, module) {
    module.doc() = "SENNA Neuro Python bindings";

    py::class_<PyNetworkHandle, std::shared_ptr<PyNetworkHandle>>(module, "NetworkHandle")
        .def(py::init<const std::string&>(), py::arg("config_path") = "configs/default.yaml")
        .def("load_sample", &PyNetworkHandle::load_sample, py::arg("image"), py::arg("label"),
             py::arg("is_train") = true)
        .def(
            "batch_train_array",
            [](PyNetworkHandle& handle, const ImageArray& images, const LabelArray& labels,
               const std::size_t ticks_per_sample) {
                const auto batch = validate_batch_arrays(images, labels);
                BatchExecutionStats stats{};
                {
                    py::gil_scoped_release release{};
                    stats = handle.batch_train_array(batch, ticks_per_sample);
                }
                return batch_result_dict(handle, stats);
            },
            py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"))
        .def(
            "batch_evaluate_array",
            [](PyNetworkHandle& handle, const ImageArray& images, const LabelArray& labels,
               const std::size_t ticks_per_sample) {
                const auto batch = validate_batch_arrays(images, labels);
                BatchExecutionStats stats{};
                {
                    py::gil_scoped_release release{};
                    stats = handle.batch_evaluate_array(batch, ticks_per_sample);
                }
                return batch_result_dict(handle, stats);
            },
            py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"))
        .def(
            "batch_train",
            [](PyNetworkHandle& handle, const std::vector<std::vector<std::uint8_t>>& images,
               const std::vector<int>& labels, const std::size_t ticks_per_sample) {
                BatchExecutionStats stats{};
                {
                    py::gil_scoped_release release{};
                    stats = handle.batch_train(images, labels, ticks_per_sample);
                }
                return batch_result_dict(handle, stats);
            },
            py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"))
        .def(
            "batch_evaluate",
            [](PyNetworkHandle& handle, const std::vector<std::vector<std::uint8_t>>& images,
               const std::vector<int>& labels, const std::size_t ticks_per_sample) {
                BatchExecutionStats stats{};
                {
                    py::gil_scoped_release release{};
                    stats = handle.batch_evaluate(images, labels, ticks_per_sample);
                }
                return batch_result_dict(handle, stats);
            },
            py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"))
        .def("step", &PyNetworkHandle::step, py::arg("n_ticks"))
        .def("step_with_trace", &PyNetworkHandle::step_with_trace, py::arg("n_ticks"))
        .def("get_prediction", &PyNetworkHandle::get_prediction)
        .def("get_lattice", &PyNetworkHandle::get_lattice)
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
        "batch_train_array",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const ImageArray& images,
           const LabelArray& labels, const std::size_t ticks_per_sample) {
            const auto batch = validate_batch_arrays(images, labels);
            BatchExecutionStats stats{};
            {
                py::gil_scoped_release release{};
                stats = handle->batch_train_array(batch, ticks_per_sample);
            }
            return batch_result_dict(*handle, stats);
        },
        py::arg("handle"), py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"));

    module.def(
        "batch_evaluate_array",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const ImageArray& images,
           const LabelArray& labels, const std::size_t ticks_per_sample) {
            const auto batch = validate_batch_arrays(images, labels);
            BatchExecutionStats stats{};
            {
                py::gil_scoped_release release{};
                stats = handle->batch_evaluate_array(batch, ticks_per_sample);
            }
            return batch_result_dict(*handle, stats);
        },
        py::arg("handle"), py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"));

    module.def(
        "batch_train",
        [](const std::shared_ptr<PyNetworkHandle>& handle,
           const std::vector<std::vector<std::uint8_t>>& images, const std::vector<int>& labels,
           const std::size_t ticks_per_sample) {
            BatchExecutionStats stats{};
            {
                py::gil_scoped_release release{};
                stats = handle->batch_train(images, labels, ticks_per_sample);
            }
            return batch_result_dict(*handle, stats);
        },
        py::arg("handle"), py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"));

    module.def(
        "batch_evaluate",
        [](const std::shared_ptr<PyNetworkHandle>& handle,
           const std::vector<std::vector<std::uint8_t>>& images, const std::vector<int>& labels,
           const std::size_t ticks_per_sample) {
            BatchExecutionStats stats{};
            {
                py::gil_scoped_release release{};
                stats = handle->batch_evaluate(images, labels, ticks_per_sample);
            }
            return batch_result_dict(*handle, stats);
        },
        py::arg("handle"), py::arg("images"), py::arg("labels"), py::arg("ticks_per_sample"));

    module.def(
        "step_with_trace",
        [](const std::shared_ptr<PyNetworkHandle>& handle, const std::size_t n_ticks) {
            return handle->step_with_trace(n_ticks);
        },
        py::arg("handle"), py::arg("n_ticks"));

    module.def(
        "get_prediction",
        [](const std::shared_ptr<PyNetworkHandle>& handle) { return handle->get_prediction(); },
        py::arg("handle"));

    module.def(
        "get_lattice",
        [](const std::shared_ptr<PyNetworkHandle>& handle) { return handle->get_lattice(); },
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
            return PyNetworkHandle::load_state(path, config_path.c_str());
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

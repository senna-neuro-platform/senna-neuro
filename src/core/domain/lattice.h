#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

#include "core/domain/neuron.h"
#include "core/domain/types.h"

namespace senna::core::domain {

struct NeighborInfo {
    NeuronId id{};
    float distance{};

    [[nodiscard]] bool operator==(const NeighborInfo&) const noexcept = default;
};

struct LatticeConfig {
    std::uint16_t width{28U};
    std::uint16_t height{28U};
    std::uint16_t depth{20U};
    float processing_density{0.70F};
    std::uint16_t output_neurons{10U};
    float excitatory_ratio{0.80F};
    float neighbor_radius{2.0F};
    NeuronConfig neuron{};
};

struct DefaultNeuronFactory {
    [[nodiscard]] Neuron operator()(const NeuronId id, const Coord3D position,
                                    const NeuronType type,
                                    const NeuronConfig& config) const noexcept {
        return Neuron{id, position, type, config};
    }
};

class Lattice final {
   public:
    explicit Lattice(const LatticeConfig config = {}) : config_(config) {
        validate_config();
        reset_storage();
    }

    template <typename RandomGenerator, typename NeuronFactory = DefaultNeuronFactory>
        requires std::uniform_random_bit_generator<RandomGenerator> &&
                 std::invocable<NeuronFactory&, NeuronId, Coord3D, NeuronType, const NeuronConfig&>
    void generate(RandomGenerator& random, NeuronFactory factory = {}) {
        clear_generation_state();
        place_sensor_layer(factory);
        place_processing_volume(random, factory);
        place_output_layer(factory);
        precompute_neighbors();
    }

    [[nodiscard]] const LatticeConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::size_t voxel_count() const noexcept { return voxels_.size(); }

    [[nodiscard]] std::size_t neuron_count() const noexcept { return neurons_.size(); }

    [[nodiscard]] std::size_t sensor_neuron_count() const noexcept { return sensor_neuron_count_; }

    [[nodiscard]] std::size_t output_neuron_count() const noexcept { return output_neuron_count_; }

    [[nodiscard]] std::size_t processing_neuron_count() const noexcept {
        return neuron_count() - sensor_neuron_count_ - output_neuron_count_;
    }

    [[nodiscard]] const std::vector<Neuron>& neurons() const noexcept { return neurons_; }

    [[nodiscard]] std::optional<NeuronId> neuron_id_at(const Coord3D position) const noexcept {
        if (!in_bounds(position)) {
            return std::nullopt;
        }

        const auto id = voxels_[flat_index(position.x, position.y, position.z)];
        if (id == kEmptyNeuronId) {
            return std::nullopt;
        }
        return id;
    }

    [[nodiscard]] bool occupied(const Coord3D position) const noexcept {
        return neuron_id_at(position).has_value();
    }

    [[nodiscard]] std::vector<NeighborInfo> neighbors(const NeuronId neuron_id) const {
        if (static_cast<std::size_t>(neuron_id) >= neurons_.size()) {
            throw std::out_of_range("NeuronId is out of lattice bounds");
        }

        if (neighbor_offsets_.size() != neurons_.size() + 1U) {
            return scan_neighbors(neuron_id, config_.neighbor_radius);
        }

        const auto begin = neighbor_offsets_[static_cast<std::size_t>(neuron_id)];
        const auto end = neighbor_offsets_[static_cast<std::size_t>(neuron_id) + 1U];
        return std::vector<NeighborInfo>(
            neighbor_data_.begin() + static_cast<std::ptrdiff_t>(begin),
            neighbor_data_.begin() + static_cast<std::ptrdiff_t>(end));
    }

    [[nodiscard]] std::vector<NeighborInfo> neighbors(const NeuronId neuron_id,
                                                      const float radius) const {
        if (static_cast<std::size_t>(neuron_id) >= neurons_.size()) {
            throw std::out_of_range("NeuronId is out of lattice bounds");
        }

        constexpr float kEpsilon = 1e-4F;
        if (std::fabs(radius - config_.neighbor_radius) <= kEpsilon) {
            return neighbors(neuron_id);
        }

        return scan_neighbors(neuron_id, radius);
    }

   private:
    static constexpr NeuronId kEmptyNeuronId = std::numeric_limits<NeuronId>::max();

    void validate_config() const {
        if (config_.width == 0U || config_.height == 0U || config_.depth < 2U) {
            throw std::invalid_argument("Lattice dimensions must be positive and depth >= 2");
        }
        if (config_.processing_density < 0.0F || config_.processing_density > 1.0F) {
            throw std::invalid_argument("processing_density must be in [0, 1]");
        }
        if (config_.excitatory_ratio < 0.0F || config_.excitatory_ratio > 1.0F) {
            throw std::invalid_argument("excitatory_ratio must be in [0, 1]");
        }
        if (config_.neighbor_radius <= 0.0F) {
            throw std::invalid_argument("neighbor_radius must be positive");
        }

        const auto output_capacity =
            static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height);
        if (config_.output_neurons == 0U ||
            static_cast<std::size_t>(config_.output_neurons) > output_capacity) {
            throw std::invalid_argument("output_neurons must be in [1, width*height]");
        }
    }

    void reset_storage() {
        voxels_.assign(total_voxels(), kEmptyNeuronId);
        neurons_.clear();
        neighbor_offsets_.clear();
        neighbor_data_.clear();
        sensor_neuron_count_ = 0U;
        output_neuron_count_ = 0U;
    }

    void clear_generation_state() {
        std::fill(voxels_.begin(), voxels_.end(), kEmptyNeuronId);
        neurons_.clear();
        neighbor_offsets_.clear();
        neighbor_data_.clear();
        sensor_neuron_count_ = 0U;
        output_neuron_count_ = 0U;
    }

    [[nodiscard]] std::size_t total_voxels() const noexcept {
        return static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height) *
               static_cast<std::size_t>(config_.depth);
    }

    [[nodiscard]] std::size_t flat_index(const std::uint16_t x, const std::uint16_t y,
                                         const std::uint16_t z) const noexcept {
        const auto width = static_cast<std::size_t>(config_.width);
        const auto height = static_cast<std::size_t>(config_.height);
        return (static_cast<std::size_t>(z) * width * height) +
               (static_cast<std::size_t>(y) * width) + static_cast<std::size_t>(x);
    }

    [[nodiscard]] bool in_bounds(const Coord3D position) const noexcept {
        return position.x < config_.width && position.y < config_.height &&
               position.z < config_.depth;
    }

    template <typename NeuronFactory>
        requires std::invocable<NeuronFactory&, NeuronId, Coord3D, NeuronType, const NeuronConfig&>
    void emplace_neuron(const Coord3D position, const NeuronType type, NeuronFactory& factory) {
        const auto index = flat_index(position.x, position.y, position.z);
        if (voxels_[index] != kEmptyNeuronId) {
            throw std::logic_error("Attempted to place a neuron into an occupied voxel");
        }

        const auto id = static_cast<NeuronId>(neurons_.size());
        voxels_[index] = id;
        neurons_.push_back(factory(id, position, type, config_.neuron));
    }

    template <typename NeuronFactory>
        requires std::invocable<NeuronFactory&, NeuronId, Coord3D, NeuronType, const NeuronConfig&>
    void place_sensor_layer(NeuronFactory& factory) {
        constexpr std::uint16_t kSensorLayerZ = 0U;
        for (std::uint16_t y = 0U; y < config_.height; ++y) {
            for (std::uint16_t x = 0U; x < config_.width; ++x) {
                emplace_neuron(Coord3D{x, y, kSensorLayerZ}, NeuronType::Excitatory, factory);
                ++sensor_neuron_count_;
            }
        }
    }

    template <typename RandomGenerator, typename NeuronFactory>
        requires std::uniform_random_bit_generator<RandomGenerator> &&
                 std::invocable<NeuronFactory&, NeuronId, Coord3D, NeuronType, const NeuronConfig&>
    void place_processing_volume(RandomGenerator& random, NeuronFactory& factory) {
        if (config_.depth <= 2U) {
            return;
        }

        std::bernoulli_distribution place_neuron(config_.processing_density);
        std::bernoulli_distribution use_excitatory(config_.excitatory_ratio);

        for (std::uint16_t z = 1U; z < static_cast<std::uint16_t>(config_.depth - 1U); ++z) {
            for (std::uint16_t y = 0U; y < config_.height; ++y) {
                for (std::uint16_t x = 0U; x < config_.width; ++x) {
                    if (!place_neuron(random)) {
                        continue;
                    }

                    const auto type =
                        use_excitatory(random) ? NeuronType::Excitatory : NeuronType::Inhibitory;
                    emplace_neuron(Coord3D{x, y, z}, type, factory);
                }
            }
        }
    }

    template <typename NeuronFactory>
        requires std::invocable<NeuronFactory&, NeuronId, Coord3D, NeuronType, const NeuronConfig&>
    void place_output_layer(NeuronFactory& factory) {
        const auto z = static_cast<std::uint16_t>(config_.depth - 1U);
        const auto layer_size =
            static_cast<std::size_t>(config_.width) * static_cast<std::size_t>(config_.height);
        std::vector<bool> used(layer_size, false);

        for (std::size_t i = 0U; i < static_cast<std::size_t>(config_.output_neurons); ++i) {
            auto linear =
                static_cast<std::size_t>(((2U * i + 1U) * layer_size) /
                                         (2U * static_cast<std::size_t>(config_.output_neurons)));

            while (linear < layer_size && used[linear]) {
                ++linear;
            }
            if (linear >= layer_size) {
                linear = layer_size - 1U;
                while (used[linear] && linear > 0U) {
                    --linear;
                }
                if (used[linear]) {
                    throw std::logic_error("Unable to place all output neurons");
                }
            }

            used[linear] = true;
            const auto x =
                static_cast<std::uint16_t>(linear % static_cast<std::size_t>(config_.width));
            const auto y =
                static_cast<std::uint16_t>(linear / static_cast<std::size_t>(config_.width));
            emplace_neuron(Coord3D{x, y, z}, NeuronType::Excitatory, factory);
            ++output_neuron_count_;
        }
    }

    [[nodiscard]] std::vector<NeighborInfo> scan_neighbors(const NeuronId neuron_id,
                                                           const float radius) const {
        if (radius <= 0.0F) {
            return {};
        }

        const auto& center = neurons_.at(static_cast<std::size_t>(neuron_id)).position();
        const auto range = static_cast<int>(std::ceil(radius));

        const auto x_min = std::max(0, static_cast<int>(center.x) - range);
        const auto y_min = std::max(0, static_cast<int>(center.y) - range);
        const auto z_min = std::max(0, static_cast<int>(center.z) - range);

        const auto x_max =
            std::min(static_cast<int>(config_.width) - 1, static_cast<int>(center.x) + range);
        const auto y_max =
            std::min(static_cast<int>(config_.height) - 1, static_cast<int>(center.y) + range);
        const auto z_max =
            std::min(static_cast<int>(config_.depth) - 1, static_cast<int>(center.z) + range);

        std::vector<NeighborInfo> result{};
        for (int z = z_min; z <= z_max; ++z) {
            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    const auto occupant = voxels_[flat_index(static_cast<std::uint16_t>(x),
                                                             static_cast<std::uint16_t>(y),
                                                             static_cast<std::uint16_t>(z))];
                    if (occupant == kEmptyNeuronId || occupant == neuron_id) {
                        continue;
                    }

                    const Coord3D other_position{static_cast<std::uint16_t>(x),
                                                 static_cast<std::uint16_t>(y),
                                                 static_cast<std::uint16_t>(z)};
                    const auto distance = center.distance(other_position);
                    if (distance <= radius) {
                        result.push_back(NeighborInfo{occupant, distance});
                    }
                }
            }
        }

        return result;
    }

    void precompute_neighbors() {
        neighbor_offsets_.assign(neurons_.size() + 1U, 0U);
        neighbor_data_.clear();

        for (std::size_t id = 0U; id < neurons_.size(); ++id) {
            neighbor_offsets_[id] = neighbor_data_.size();
            const auto local_neighbors =
                scan_neighbors(static_cast<NeuronId>(id), config_.neighbor_radius);
            neighbor_data_.insert(neighbor_data_.end(), local_neighbors.begin(),
                                  local_neighbors.end());
        }

        neighbor_offsets_[neurons_.size()] = neighbor_data_.size();
    }

    LatticeConfig config_{};
    std::vector<NeuronId> voxels_{};
    std::vector<Neuron> neurons_{};

    std::vector<std::size_t> neighbor_offsets_{};
    std::vector<NeighborInfo> neighbor_data_{};

    std::size_t sensor_neuron_count_{0U};
    std::size_t output_neuron_count_{0U};
};

}  // namespace senna::core::domain

#pragma once

#include <cmath>
#include <cstdint>

namespace senna::core::domain {

using NeuronId = std::uint32_t;
using SynapseId = std::uint32_t;
using Time = float;
using Voltage = float;
using Weight = float;

enum class NeuronType : std::uint8_t {
    Excitatory = 0,
    Inhibitory = 1,
};

template <typename CoordinateT>
struct Coord3 {
    CoordinateT x{};
    CoordinateT y{};
    CoordinateT z{};

    template <typename OtherCoordinateT>
    [[nodiscard]] float distance(const Coord3<OtherCoordinateT>& other) const noexcept {
        const auto dx = static_cast<float>(x) - static_cast<float>(other.x);
        const auto dy = static_cast<float>(y) - static_cast<float>(other.y);
        const auto dz = static_cast<float>(z) - static_cast<float>(other.z);
        return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
    }
};

using Coord3D = Coord3<std::uint16_t>;

struct SpikeEvent {
    NeuronId source{};
    NeuronId target{};
    Time arrival{};
    Weight value{};

    [[nodiscard]] bool operator<(const SpikeEvent& other) const noexcept {
        return arrival < other.arrival;
    }

    [[nodiscard]] bool operator>(const SpikeEvent& other) const noexcept { return other < *this; }
};

template <typename EventT>
struct ArrivalEarlier {
    [[nodiscard]] bool operator()(const EventT& lhs, const EventT& rhs) const noexcept {
        return lhs.arrival < rhs.arrival;
    }
};

}  // namespace senna::core::domain

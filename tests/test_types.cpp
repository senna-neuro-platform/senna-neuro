#include <cmath>
#include <cstdint>
#include <iostream>

#include "core/domain/types.h"

namespace {

bool almost_equal(const float lhs, const float rhs, const float eps = 1e-5F) {
    return std::fabs(lhs - rhs) <= eps;
}

int run_distance_tests() {
    using senna::core::domain::Coord3;
    using senna::core::domain::Coord3D;

    const Coord3D origin{0, 0, 0};
    const Coord3D point{3, 4, 0};
    const Coord3D point_3d{2, 3, 6};
    const Coord3<int> int_point{3, 4, 0};

    if (!almost_equal(origin.distance(point), 5.0F)) {
        std::cerr << "distance(0,0,0 -> 3,4,0) must be 5.0\n";
        return 1;
    }

    if (!almost_equal(origin.distance(origin), 0.0F)) {
        std::cerr << "distance to self must be 0.0\n";
        return 1;
    }

    if (!almost_equal(origin.distance(point_3d), 7.0F)) {
        std::cerr << "distance(0,0,0 -> 2,3,6) must be 7.0\n";
        return 1;
    }

    if (!almost_equal(origin.distance(int_point), 5.0F)) {
        std::cerr << "distance between Coord3<uint16_t> and Coord3<int> must be 5.0\n";
        return 1;
    }

    return 0;
}

int run_spike_event_tests() {
    using senna::core::domain::ArrivalEarlier;
    using senna::core::domain::SpikeEvent;

    const SpikeEvent early{1U, 2U, 1.0F, 0.25F};
    const SpikeEvent late{1U, 2U, 2.0F, 0.25F};
    const ArrivalEarlier<SpikeEvent> compare_by_arrival{};

    if (!(early < late)) {
        std::cerr << "early event must be < late event\n";
        return 1;
    }

    if (late < early) {
        std::cerr << "late event must not be < early event\n";
        return 1;
    }

    if (!(late > early)) {
        std::cerr << "late event must be > early event\n";
        return 1;
    }

    if (!compare_by_arrival(early, late)) {
        std::cerr << "ArrivalEarlier must return true for early/late pair\n";
        return 1;
    }

    if (compare_by_arrival(late, early)) {
        std::cerr << "ArrivalEarlier must return false for late/early pair\n";
        return 1;
    }

    return 0;
}

int run_neuron_type_tests() {
    using senna::core::domain::NeuronType;

    if (NeuronType::Excitatory == NeuronType::Inhibitory) {
        std::cerr << "Excitatory and Inhibitory types must differ\n";
        return 1;
    }

    if (static_cast<std::uint8_t>(NeuronType::Excitatory) ==
        static_cast<std::uint8_t>(NeuronType::Inhibitory)) {
        std::cerr << "Underlying enum values must differ\n";
        return 1;
    }

    return 0;
}

}  // namespace

int main() {
    if (const int rc = run_distance_tests(); rc != 0) {
        return rc;
    }

    if (const int rc = run_spike_event_tests(); rc != 0) {
        return rc;
    }

    if (const int rc = run_neuron_type_tests(); rc != 0) {
        return rc;
    }

    return 0;
}

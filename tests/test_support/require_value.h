#pragma once

#include <optional>
#include <stdexcept>

template <typename T>
T require_value(const std::optional<T>& value, const char* message) {
    if (!value.has_value()) {
        throw std::runtime_error(message);
    }
    return *value;
}

cmake_minimum_required(VERSION 3.25)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SENNA_VERSION_FULL)
string(STRIP "${SENNA_VERSION_FULL}" SENNA_VERSION_FULL)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" SENNA_VERSION_CORE "${SENNA_VERSION_FULL}")
if(NOT SENNA_VERSION_CORE)
    message(FATAL_ERROR "VERSION must start with numeric A.B.C, got: ${SENNA_VERSION_FULL}")
endif()

project(senna_neuro VERSION ${SENNA_VERSION_CORE} LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SENNA_ENABLE_SANITIZERS "Enable ASan and UBSan for Debug builds" OFF)

add_library(senna_domain INTERFACE)
target_include_directories(senna_domain INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(senna_simulator src/main.cpp)
target_link_libraries(senna_simulator PRIVATE senna_domain)
target_compile_definitions(senna_simulator PRIVATE SENNA_VERSION="${SENNA_VERSION_FULL}")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_simulator PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(senna_simulator PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

if(BUILD_TESTING)
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")
    find_package(GTest REQUIRED MODULE)
    include(GoogleTest)

    add_executable(test_types tests/test_types.cpp)
    target_link_libraries(test_types PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_types PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_types PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_types PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_types)

    add_executable(test_neuron tests/test_neuron.cpp)
    target_link_libraries(test_neuron PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_neuron PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_neuron PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_neuron PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_neuron)

    add_executable(test_synapse tests/test_synapse.cpp)
    target_link_libraries(test_synapse PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_synapse PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_synapse PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_synapse PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_synapse)

    add_executable(test_lattice tests/test_lattice.cpp)
    target_link_libraries(test_lattice PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_lattice PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_lattice PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_lattice PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_lattice)

    add_executable(test_event_queue tests/test_event_queue.cpp)
    target_link_libraries(test_event_queue PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_event_queue PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_event_queue PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_event_queue PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_event_queue)

    add_executable(test_network_builder tests/test_network_builder.cpp)
    target_link_libraries(test_network_builder PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_network_builder PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_network_builder PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_network_builder PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_network_builder)

    add_executable(test_io tests/test_io.cpp)
    target_link_libraries(test_io PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_io PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_io PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_io PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_io)
endif()

install(TARGETS senna_simulator RUNTIME DESTINATION bin)

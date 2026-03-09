cmake_minimum_required(VERSION 3.25)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" SENNA_VERSION_FULL)
string(STRIP "${SENNA_VERSION_FULL}" SENNA_VERSION_FULL)
string(REGEX MATCH "^([0-9]+\\.[0-9]+\\.[0-9]+)" SENNA_VERSION_CORE "${SENNA_VERSION_FULL}")
if(NOT SENNA_VERSION_CORE)
    message(FATAL_ERROR "VERSION must start with numeric A.B.C, got: ${SENNA_VERSION_FULL}")
endif()

project(senna_neuro VERSION ${SENNA_VERSION_CORE} LANGUAGES CXX)

include(CTest)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SENNA_ENABLE_SANITIZERS "Enable ASan and UBSan for Debug builds" OFF)
option(SENNA_BUILD_PYTHON_BINDINGS "Build pybind11 module senna_core" ON)

set(SENNA_DOMAIN_SOURCES
    src/core/domain/neuron.cpp
    src/core/domain/synapse.cpp
    src/core/engine/event_queue.cpp
    src/core/engine/network_builder.cpp
    src/core/engine/simulation_engine.cpp
    src/core/engine/time_manager.cpp
    src/core/io/first_spike_decoder.cpp
    src/core/io/rate_encoder.cpp
    src/core/metrics/metrics_collector.cpp
    src/core/plasticity/homeostasis.cpp
    src/core/plasticity/stdp.cpp
    src/core/plasticity/structural_plasticity.cpp
    src/core/plasticity/supervisor.cpp
)

add_library(senna_domain STATIC ${SENNA_DOMAIN_SOURCES})
target_include_directories(senna_domain PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
set_target_properties(senna_domain PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_domain PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_domain PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

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

if(SENNA_BUILD_PYTHON_BINDINGS OR BUILD_TESTING)
    find_package(HDF5 REQUIRED MODULE)
endif()

if(SENNA_BUILD_PYTHON_BINDINGS)
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 REQUIRED CONFIG)
    find_package(yaml-cpp REQUIRED CONFIG)

    pybind11_add_module(senna_core MODULE src/bindings/python_module.cpp)
    target_link_libraries(senna_core PRIVATE senna_domain yaml-cpp::yaml-cpp HDF5::HDF5)
    set_target_properties(senna_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(senna_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(senna_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(senna_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
endif()

set(SENNA_PYTHON_TEST_ENV "PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}/python")
if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE SENNA_LIBASAN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(SENNA_LIBASAN_PATH AND NOT SENNA_LIBASAN_PATH STREQUAL "libasan.so")
        list(APPEND SENNA_PYTHON_TEST_ENV
            "LD_PRELOAD=${SENNA_LIBASAN_PATH}"
            "ASAN_OPTIONS=detect_leaks=0"
            "UBSAN_OPTIONS=print_stacktrace=1"
        )
    else()
        message(WARNING "Unable to resolve libasan.so for sanitized Python integration tests")
    endif()
endif()

if(BUILD_TESTING)
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}")
    find_package(GTest REQUIRED MODULE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
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

    add_executable(test_stdp tests/test_stdp.cpp)
    target_link_libraries(test_stdp PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_stdp PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_stdp PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_stdp PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_stdp)

    add_executable(test_supervisor tests/test_supervisor.cpp)
    target_link_libraries(test_supervisor PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_supervisor PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_supervisor PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_supervisor PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_supervisor)

    add_executable(test_homeostasis tests/test_homeostasis.cpp)
    target_link_libraries(test_homeostasis PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_homeostasis PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_homeostasis PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_homeostasis PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_homeostasis)

    add_executable(test_structural_plasticity tests/test_structural_plasticity.cpp)
    target_link_libraries(test_structural_plasticity PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_structural_plasticity PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_structural_plasticity PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_structural_plasticity PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_structural_plasticity)

    add_executable(test_persistence tests/test_persistence.cpp)
    target_link_libraries(test_persistence PRIVATE senna_domain GTest::gtest_main HDF5::HDF5)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_persistence PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_persistence PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_persistence PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_persistence)

    add_executable(test_metrics tests/test_metrics.cpp)
    target_link_libraries(test_metrics PRIVATE senna_domain GTest::gtest_main)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_metrics PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(test_metrics PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(test_metrics PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()

    gtest_discover_tests(test_metrics)

    add_test(
        NAME test_prometheus_exporter_format
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/infra/simulator/test_simulator_server.py
    )

    if(SENNA_BUILD_PYTHON_BINDINGS)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pytest --version
            RESULT_VARIABLE SENNA_PYTEST_STATUS
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(SENNA_PYTEST_STATUS EQUAL 0)
            add_test(
                NAME test_python_bindings_integration
                COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_CURRENT_SOURCE_DIR}/python/tests/test_integration.py
            )
            set_tests_properties(
                test_python_bindings_integration
                PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    ENVIRONMENT "${SENNA_PYTHON_TEST_ENV}"
            )
        else()
            message(STATUS "pytest not found, skipping test_python_bindings_integration")
        endif()
    endif()
endif()

install(TARGETS senna_simulator RUNTIME DESTINATION bin)
if(SENNA_BUILD_PYTHON_BINDINGS)
    install(TARGETS senna_core LIBRARY DESTINATION python)
endif()

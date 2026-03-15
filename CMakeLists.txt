cmake_minimum_required(VERSION 3.25)

project(senna-neuro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SENNA_ENABLE_SANITIZERS "Enable ASan/UBSan" OFF)

if(SENNA_ENABLE_SANITIZERS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

include(CTest)
find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS system json)
find_package(gRPC REQUIRED)
find_package(protobuf REQUIRED)
# Ensure Conan protobuf headers take priority over any system-installed protobuf.
include_directories(BEFORE SYSTEM "${protobuf_INCLUDE_DIR}")

# --- Proto generation ---
set(PROTO_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/proto/senna.proto")
set(PROTO_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_OUT}")

set(PROTO_GENERATED_SRCS
  "${PROTO_OUT}/senna.pb.cc"
  "${PROTO_OUT}/senna.pb.h"
  "${PROTO_OUT}/senna.grpc.pb.cc"
  "${PROTO_OUT}/senna.grpc.pb.h"
)

add_custom_command(
  OUTPUT ${PROTO_GENERATED_SRCS}
  COMMAND "${protobuf_INCLUDE_DIR}/../bin/protoc"
    --proto_path="${CMAKE_CURRENT_SOURCE_DIR}/src/proto"
    --proto_path="${protobuf_INCLUDE_DIR}"
    --cpp_out="${PROTO_OUT}"
    --grpc_out="${PROTO_OUT}"
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    "${PROTO_SRC}"
  DEPENDS "${PROTO_SRC}"
  COMMENT "Generating protobuf/gRPC C++ sources"
)

add_library(senna_proto ${PROTO_GENERATED_SRCS})
target_include_directories(senna_proto PUBLIC "${PROTO_OUT}")
target_link_libraries(senna_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
set_target_properties(senna_proto PROPERTIES
  CXX_STANDARD 23
  CXX_CLANG_TIDY ""
)
# Suppress warnings in generated code.
target_compile_options(senna_proto PRIVATE -w)

set(SENNA_CORE_SOURCES
  src/core/spatial/lattice.cpp
  src/core/spatial/neighbor_index.cpp
  src/core/neural/neuron.cpp
  src/core/neural/neuron_pool.cpp
  src/core/decoding/first_spike_decoder.cpp
  src/core/synaptic/synapse.cpp
  src/core/synaptic/synapse_index.cpp
  src/core/temporal/event_queue.cpp
  src/core/temporal/time_manager.cpp
  src/core/plasticity/stdp.cpp
  src/core/plasticity/stdp_worker.cpp
  src/core/plasticity/homeostasis.cpp
  src/core/plasticity/structural.cpp
  src/core/encoding/rate_encoder.cpp
  src/core/encoding/wta_decoder.cpp
  src/core/config/runtime_config.cpp
  src/core/observability/metrics_collector.cpp
  src/core/observability/prometheus_exporter.cpp
  src/core/network/network_builder.cpp
  src/core/network/spike_loop.cpp
  src/core/interfaces/grpc_server.cpp
  src/core/interfaces/ws_server.cpp
)

add_library(senna_core_lib ${SENNA_CORE_SOURCES})
target_compile_features(senna_core_lib PUBLIC cxx_std_23)
target_include_directories(senna_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(ZLIB REQUIRED)

target_link_libraries(senna_core_lib PUBLIC yaml-cpp::yaml-cpp Boost::system Boost::json ZLIB::ZLIB senna_proto)

add_executable(senna_core src/core/main.cpp)
target_compile_features(senna_core PRIVATE cxx_std_23)
target_include_directories(senna_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(senna_core PRIVATE senna_core_lib Threads::Threads)

add_library(senna_trainer_lib
  src/trainer/mnist_loader.cpp
  src/trainer/training_pipeline.cpp
  src/trainer/config.cpp
)
target_compile_features(senna_trainer_lib PUBLIC cxx_std_23)
target_include_directories(senna_trainer_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(senna_trainer_lib PUBLIC senna_proto yaml-cpp::yaml-cpp Threads::Threads)

add_executable(senna_trainer src/trainer/main.cpp)
target_compile_features(senna_trainer PRIVATE cxx_std_23)
target_include_directories(senna_trainer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(senna_trainer PRIVATE senna_trainer_lib)

# --- Tests ---
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(test_spatial
    tests/spatial/lattice_test.cpp
    tests/spatial/neighbor_index_test.cpp
  )
  target_link_libraries(test_spatial PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_spatial)

  add_executable(test_neural
    tests/neural/neuron_pool_test.cpp
  )
  target_link_libraries(test_neural PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_neural)

  add_executable(test_synaptic
    tests/synaptic/synapse_index_test.cpp
  )
  target_link_libraries(test_synaptic PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_synaptic)

  add_executable(test_plasticity
    tests/plasticity/stdp_test.cpp
    tests/plasticity/stdp_worker_test.cpp
    tests/plasticity/homeostasis_test.cpp
    tests/plasticity/structural_test.cpp
  )
  target_link_libraries(test_plasticity PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_plasticity)

  add_executable(test_temporal
    tests/temporal/event_queue_test.cpp
  )
  target_link_libraries(test_temporal PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_temporal)

  add_executable(test_encoding
    tests/encoding/rate_encoder_test.cpp  
  )
  target_link_libraries(test_encoding PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_encoding)

  add_executable(test_decoding
    tests/decoding/first_spike_decoder_test.cpp
  )
  target_link_libraries(test_decoding PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_decoding)

  add_executable(test_observability
    tests/observability/metrics_observability_test.cpp
  )
  target_link_libraries(test_observability PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_observability)

  add_executable(test_interfaces
    tests/interfaces/ws_server_test.cpp
  )
  target_link_libraries(test_interfaces PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_interfaces)

  add_executable(test_grpc
    tests/interfaces/grpc_server_test.cpp
  )
  target_link_libraries(test_grpc PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_grpc)

  add_executable(test_smoke
    tests/integration/lattice_smoke_test.cpp
    tests/integration/network_smoke_test.cpp
  )
  target_link_libraries(test_smoke PRIVATE senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_smoke)

  add_executable(test_trainer
    tests/trainer/mnist_loader_test.cpp
    tests/trainer/config_test.cpp
    tests/trainer/training_pipeline_test.cpp
  )
  target_link_libraries(test_trainer PRIVATE senna_trainer_lib senna_core_lib GTest::gtest_main Threads::Threads)
  gtest_discover_tests(test_trainer)
endif()

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

add_executable(senna_simulator src/main.cpp)
target_compile_definitions(senna_simulator PRIVATE SENNA_VERSION="${SENNA_VERSION_FULL}")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_simulator PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(SENNA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(senna_simulator PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(senna_simulator PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

install(TARGETS senna_simulator RUNTIME DESTINATION bin)

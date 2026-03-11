from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout


class SennaNeuroConan(ConanFile):
    name = "senna-neuro"
    version = "0.2.0-dev"
    package_type = "application"

    settings = "os", "arch", "compiler", "build_type"

    requires = (
        "boost/1.84.0",
        "grpc/1.62.2",
        "protobuf/4.25.3",
        "spdlog/1.14.1",
        "fmt/10.2.1",
        "yaml-cpp/0.8.0",
        "gtest/1.14.0",
    )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout


class SennaNeuroConan(ConanFile):
    name = "senna-neuro"
    version = "0.2.2-dev"
    package_type = "application"

    settings = "os", "arch", "compiler", "build_type"

    requires = (
        "boost/1.88.0",
        "grpc/1.78.1",
        "spdlog/1.14.1",
        "fmt/10.2.1",
        "yaml-cpp/0.8.0",
        "gtest/1.17.0",
    )

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
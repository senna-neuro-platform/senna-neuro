from conan import ConanFile
from pathlib import Path


class SennaNeuroConan(ConanFile):
    name = "senna-neuro"
    version = Path(__file__).resolve().parent.joinpath("VERSION").read_text(encoding="utf-8").strip()
    package_type = "application"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    requires = (
        "pybind11/2.13.6",
        "hdf5/1.14.5",
        "spdlog/1.14.1",
        "fmt/10.2.1",
        "gtest/1.15.0",
        "yaml-cpp/0.8.0",
    )

# ADR-0011: Python Bindings and Training Contract

- Status: Accepted
- Date: 2026-03-07

## Context

Adds a Python layer over the C++ core, so a stable API is required for training, metrics, save/load operations, and robustness checks.

## Decision

1. The Python API contract is fixed in the pybind11 module `senna_core` (`src/bindings/python_module.cpp`).
2. Required operations in the contract: `create_network`, `load_sample`, `step`, `get_prediction`, `get_metrics`, `save_state`, `load_state`, `inject_noise`, `remove_neurons`.
3. Training supervision is implemented as a separate `supervise(expected_label)` operation through a teacher spike on the correct output neuron.
4. YAML configuration is validated only on the C++ side (through `yaml-cpp`); the Python layer does not duplicate validation.
5. The Python training pipeline (`python/senna/training.py`, `python/train.py`) uses the module contract, reads real MNIST locally through host-installed `torch` and `torchvision` from `data/MNIST/raw`, and allows a synthetic dataset only with an explicit `--dataset synthetic`.
6. Python integration tests are executed through CTest with `pytest` when `pytest` is available.

## Consequences

- The interface between C++ and Python becomes explicit and testable.
- Major configuration errors are detected at the C++ boundary before training starts.
- The training script and tests can be run locally and in CI in one consistent way through CMake and CTest.

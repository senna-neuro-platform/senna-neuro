SHELL := /bin/bash

CONAN ?= conan
CMAKE ?= cmake
CTEST ?= ctest
PYTHON ?= python3
DOCKER_COMPOSE ?= docker compose
CONAN_INSTALL_FLAGS := --build=missing -c tools.cmake.cmaketoolchain:user_presets= -cc core:skip_warnings='["deprecated"]'
CONAN_PROFILE_DIR := build/conan/profiles
CONAN_HOST_PROFILE := $(CONAN_PROFILE_DIR)/host
CONAN_BUILD_PROFILE := $(CONAN_PROFILE_DIR)/build
CONAN_PROFILE_ARGS := --profile:host=$(CONAN_HOST_PROFILE) --profile:build=$(CONAN_BUILD_PROFILE)
LOCAL_PYTHON_PACKAGES := .python-packages
PROJECT_PYTHONPATH := $(LOCAL_PYTHON_PACKAGES)$(if $(PYTHONPATH),:$(PYTHONPATH),)
MNIST_DIR := data/MNIST/raw
MNIST_URL_BASE := https://storage.googleapis.com/cvdf-datasets/mnist
MNIST_FILES := train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

CPP_FILES := $(shell find src tests -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \) 2>/dev/null)
PYTHON_LINT_TARGETS := python/senna python/train.py infra scripts docs/acceptance/scripts

.PHONY: help conan-setup python-setup data-mnist install fmt lint build-debug build-release build-sanitize test up up-build down logs e2e-smoke

help:
	@echo "Available targets:"
	@echo "  make install         - install Conan deps, project-local Python deps, and MNIST data"
	@echo "  make python-setup    - install required host Python packages into .python-packages"
	@echo "  make data-mnist      - download MNIST dataset into data/MNIST/raw"
	@echo "  make lint            - run production-code formatting and linters"
	@echo "  make build-debug     - configure and build Debug preset"
	@echo "  make build-release   - configure and build Release preset"
	@echo "  make build-sanitize  - configure and build Sanitize preset"
	@echo "  make test            - run tests (ctest debug)"
	@echo "  make up              - docker compose up -d --force-recreate"
	@echo "  make up-build        - docker compose build runtime images and recreate services"
	@echo "  make down            - docker compose down"
	@echo "  make logs            - docker compose logs"
	@echo "  make e2e-smoke       - run lightweight deployment-to-training smoke validation"

conan-setup:
	@mkdir -p $(CONAN_PROFILE_DIR)
	@compiler_version=$$(g++ -dumpfullversion -dumpversion | cut -d. -f1); \
	printf "[settings]\nos=Linux\narch=x86_64\ncompiler=gcc\ncompiler.version=%s\ncompiler.libcxx=libstdc++11\ncompiler.cppstd=gnu23\n" "$$compiler_version" > $(CONAN_HOST_PROFILE)
	@compiler_version=$$(g++ -dumpfullversion -dumpversion | cut -d. -f1); \
	printf "[settings]\nos=Linux\narch=x86_64\nbuild_type=Release\ncompiler=gcc\ncompiler.version=%s\ncompiler.libcxx=libstdc++11\ncompiler.cppstd=gnu23\n" "$$compiler_version" > $(CONAN_BUILD_PROFILE)
	@if ! $(CONAN) remote list | grep -q '^conancenter:'; then \
		$(CONAN) remote add conancenter https://center2.conan.io; \
	fi

data-mnist:
	@mkdir -p $(MNIST_DIR)
	@for file in $(MNIST_FILES); do \
		if [ -f "$(MNIST_DIR)/$$file" ]; then \
			echo "MNIST $$file already exists"; \
			continue; \
		fi; \
		url="$(MNIST_URL_BASE)/$$file.gz"; \
		gz_file="$(MNIST_DIR)/$$file.gz"; \
		echo "Downloading $$url"; \
		if command -v curl >/dev/null 2>&1; then \
			curl -fsSL --retry 3 --retry-delay 1 "$$url" -o "$$gz_file"; \
		elif command -v wget >/dev/null 2>&1; then \
			wget -q -O "$$gz_file" "$$url"; \
		else \
			echo "Neither curl nor wget is available to download MNIST"; \
			exit 1; \
		fi; \
		python3 -c 'import gzip, shutil, sys; src, dst = sys.argv[1], sys.argv[2]; f_in = gzip.open(src, "rb"); f_out = open(dst, "wb"); shutil.copyfileobj(f_in, f_out); f_in.close(); f_out.close()' "$$gz_file" "$(MNIST_DIR)/$$file"; \
		rm -f "$$gz_file"; \
		echo "Saved $(MNIST_DIR)/$$file"; \
	done

python-setup:
	$(PYTHON) scripts/setup_python_packages.py --target $(LOCAL_PYTHON_PACKAGES)

install: conan-setup python-setup data-mnist
	$(CONAN) install . --output-folder=build/debug $(CONAN_INSTALL_FLAGS) $(CONAN_PROFILE_ARGS) -s:h build_type=Debug

build-debug: install
	$(CMAKE) --preset debug
	$(CMAKE) --build --preset debug

build-release:
	$(CONAN) install . --output-folder=build/release $(CONAN_INSTALL_FLAGS) $(CONAN_PROFILE_ARGS) -s:h build_type=Release
	$(CMAKE) --preset release
	$(CMAKE) --build --preset release

build-sanitize:
	$(CONAN) install . --output-folder=build/sanitize $(CONAN_INSTALL_FLAGS) $(CONAN_PROFILE_ARGS) -s:h build_type=Debug
	$(CMAKE) --preset sanitize
	$(CMAKE) --build --preset sanitize

test: build-debug
	$(CTEST) --preset debug

fmt:
	@if [ -n "$(CPP_FILES)" ]; then \
		clang-format -i $(CPP_FILES); \
	fi
	env PYTHONPATH="$(PROJECT_PYTHONPATH)" $(PYTHON) -m ruff format .

lint: fmt
	$(MAKE) build-debug
	$(PYTHON) scripts/run_clang_tidy.py --build-dir build/debug --paths src
	env PYTHONPATH="$(PROJECT_PYTHONPATH)" $(PYTHON) -m ruff check $(PYTHON_LINT_TARGETS)

up:
	$(DOCKER_COMPOSE) up -d --force-recreate --no-build

up-build:
	$(DOCKER_COMPOSE) build simulator artifact-uploader
	$(DOCKER_COMPOSE) up -d --force-recreate --no-build

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs

e2e-smoke:
	bash docs/acceptance/scripts/run_e2e_smoke.sh

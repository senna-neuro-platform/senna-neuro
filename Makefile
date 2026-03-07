SHELL := /bin/bash

CONAN ?= conan
CMAKE ?= cmake
CTEST ?= ctest
DOCKER_COMPOSE ?= docker compose
CONAN_INSTALL_FLAGS := --build=missing -c tools.cmake.cmaketoolchain:user_presets= -cc core:skip_warnings='["deprecated"]'
CONAN_PROFILE_DIR := build/conan/profiles
CONAN_HOST_PROFILE := $(CONAN_PROFILE_DIR)/host
CONAN_BUILD_PROFILE := $(CONAN_PROFILE_DIR)/build
CONAN_PROFILE_ARGS := --profile:host=$(CONAN_HOST_PROFILE) --profile:build=$(CONAN_BUILD_PROFILE)

CPP_FILES := $(shell find src tests -type f \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hh' -o -name '*.hpp' -o -name '*.hxx' \) 2>/dev/null)

.PHONY: help conan-setup install fmt lint build-debug build-release build-sanitize build-saniryze test up down logs

help:
	@echo "Available targets:"
	@echo "  make install         - install Conan dependencies for debug profile"
	@echo "  make lint            - run formatting and linters (fmt + lint)"
	@echo "  make build-debug     - configure and build Debug preset"
	@echo "  make build-release   - configure and build Release preset"
	@echo "  make build-sanitize  - configure and build Sanitize preset"
	@echo "  make test            - run tests (ctest debug)"
	@echo "  make up              - docker compose up -d"
	@echo "  make down            - docker compose down"
	@echo "  make logs            - docker compose logs"

conan-setup:
	@mkdir -p $(CONAN_PROFILE_DIR)
	@compiler_version=$$(g++ -dumpfullversion -dumpversion | cut -d. -f1); \
	printf "[settings]\nos=Linux\narch=x86_64\ncompiler=gcc\ncompiler.version=%s\ncompiler.libcxx=libstdc++11\ncompiler.cppstd=gnu23\n" "$$compiler_version" > $(CONAN_HOST_PROFILE)
	@compiler_version=$$(g++ -dumpfullversion -dumpversion | cut -d. -f1); \
	printf "[settings]\nos=Linux\narch=x86_64\nbuild_type=Release\ncompiler=gcc\ncompiler.version=%s\ncompiler.libcxx=libstdc++11\ncompiler.cppstd=gnu17\n" "$$compiler_version" > $(CONAN_BUILD_PROFILE)
	@if ! $(CONAN) remote list | grep -q '^conancenter:'; then \
		$(CONAN) remote add conancenter https://center2.conan.io; \
	fi

install: conan-setup
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
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format .; \
	fi

lint: build-debug fmt
	clang-tidy src/main.cpp -p build/debug
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check .; \
	else \
		echo "ruff not found: skipping Python lint"; \
	fi

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs

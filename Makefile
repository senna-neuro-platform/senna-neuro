# ──────────────────────────────────────────────────────────────
#  SENNA Neuro — Makefile
# ──────────────────────────────────────────────────────────────
.POSIX:
SHELL       := /bin/bash
.SHELLFLAGS := -euo pipefail -c
MAKEFLAGS   += --warn-undefined-variables --no-builtin-rules

.DEFAULT_GOAL := help

# ── Project metadata ─────────────────────────────────────────
VERSION     := $(shell cat VERSION 2>/dev/null || echo "0.0.0")
COMPOSE_FILE := docker-compose.yml

# ── Tool overrides ───────────────────────────────────────────
CONAN        ?= conan
CMAKE        ?= cmake
CTEST        ?= ctest
CLANG_FORMAT ?= clang-format
CLANG_TIDY   ?= clang-tidy
DOCKER       ?= docker

# ── Directories ──────────────────────────────────────────────
BUILD_DEBUG    := build/debug
BUILD_RELEASE  := build/release
BUILD_SANITIZE := build/sanitize
SRC_DIRS       := src tests

# ── Docker images ────────────────────────────────────────────
REGISTRY     ?=
IMAGE_PREFIX ?= senna-neuro
CORE_IMAGE      := $(REGISTRY)$(IMAGE_PREFIX)-core:$(VERSION)
TRAINER_IMAGE   := $(REGISTRY)$(IMAGE_PREFIX)-trainer:$(VERSION)
VISUALIZER_IMAGE := $(REGISTRY)$(IMAGE_PREFIX)-visualizer:$(VERSION)

# ── Source file lists (lazy-evaluated) ───────────────────────
FORMAT_FILES = $(shell find $(SRC_DIRS) -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null)
TIDY_FILES   = $(shell find $(SRC_DIRS) -type f -name '*.cpp' 2>/dev/null)

# ── Sentinel files (avoid redundant reconfiguration) ─────────
CONFIGURED_DEBUG    := $(BUILD_DEBUG)/.configured
CONFIGURED_RELEASE  := $(BUILD_RELEASE)/.configured
CONFIGURED_SANITIZE := $(BUILD_SANITIZE)/.configured

# ══════════════════════════════════════════════════════════════
#  Help
# ══════════════════════════════════════════════════════════════
.PHONY: help
help: ## Show this help
	@printf '\nUsage: make \033[36m<target>\033[0m\n\n'
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo

# ══════════════════════════════════════════════════════════════
#  Configure (Conan + CMake)
# ══════════════════════════════════════════════════════════════
.PHONY: configure-debug configure-release configure-sanitize

$(CONFIGURED_DEBUG): conanfile.py CMakeLists.txt CMakePresets.json
	@echo "── Configuring debug ──"
	$(CONAN) install . --output-folder=$(BUILD_DEBUG) --build=missing -s build_type=Debug
	$(CMAKE) --preset debug
	@touch $@

$(CONFIGURED_RELEASE): conanfile.py CMakeLists.txt CMakePresets.json
	@echo "── Configuring release ──"
	$(CONAN) install . --output-folder=$(BUILD_RELEASE) --build=missing -s build_type=Release
	$(CMAKE) --preset release
	@touch $@

$(CONFIGURED_SANITIZE): conanfile.py CMakeLists.txt CMakePresets.json
	@echo "── Configuring sanitize ──"
	$(CONAN) install . --output-folder=$(BUILD_SANITIZE) --build=missing -s build_type=Debug
	$(CMAKE) --preset sanitize
	@touch $@

configure-debug: $(CONFIGURED_DEBUG)       ## Configure debug build
configure-release: $(CONFIGURED_RELEASE)   ## Configure release build
configure-sanitize: $(CONFIGURED_SANITIZE) ## Configure sanitize build

# ══════════════════════════════════════════════════════════════
#  Build
# ══════════════════════════════════════════════════════════════
.PHONY: build build-debug build-release build-sanitize

build: build-debug ## Build (alias for build-debug)

build-debug: $(CONFIGURED_DEBUG) ## Build debug
	$(CMAKE) --build --preset debug

build-release: $(CONFIGURED_RELEASE) ## Build release
	$(CMAKE) --build --preset release

build-sanitize: $(CONFIGURED_SANITIZE) ## Build sanitize (ASan+UBSan)
	$(CMAKE) --build --preset sanitize

# ══════════════════════════════════════════════════════════════
#  Test
# ══════════════════════════════════════════════════════════════
.PHONY: test test-release test-sanitize

test: build-debug ## Run tests (debug)
	$(CTEST) --preset debug

test-release: build-release ## Run tests (release)
	$(CTEST) --preset release

test-sanitize: build-sanitize ## Run tests (sanitize)
	$(CTEST) --preset sanitize

# ══════════════════════════════════════════════════════════════
#  Lint & Format
# ══════════════════════════════════════════════════════════════
.PHONY: lint fmt-check fmt tidy

lint: fmt-check tidy ## Run all linters

fmt-check: ## Check formatting (clang-format --dry-run)
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No C/C++ source files found."; \
	else \
		$(CLANG_FORMAT) --dry-run --Werror $(FORMAT_FILES); \
	fi

fmt: ## Auto-format sources in-place
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No C/C++ source files found."; \
	else \
		$(CLANG_FORMAT) -i $(FORMAT_FILES); \
		echo "Formatted $$(echo $(FORMAT_FILES) | wc -w) files."; \
	fi

tidy: $(CONFIGURED_DEBUG) ## Run clang-tidy
	@if [ -z "$(TIDY_FILES)" ]; then \
		echo "No C/C++ source files found."; \
	else \
		$(CLANG_TIDY) -p $(BUILD_DEBUG) $(TIDY_FILES); \
	fi

# ══════════════════════════════════════════════════════════════
#  Docker Compose
# ══════════════════════════════════════════════════════════════
.PHONY: up down restart logs ps

up: ## Start all services
	$(DOCKER) compose -f $(COMPOSE_FILE) up -d

down: ## Stop all services
	$(DOCKER) compose -f $(COMPOSE_FILE) down

restart: down up ## Restart all services

logs: ## Tail service logs
	$(DOCKER) compose -f $(COMPOSE_FILE) logs -f --tail=200

ps: ## Show running services
	$(DOCKER) compose -f $(COMPOSE_FILE) ps

# ══════════════════════════════════════════════════════════════
#  Docker Images
# ══════════════════════════════════════════════════════════════
.PHONY: docker-build docker-push

docker-build: ## Build all Docker images
	$(DOCKER) build -t $(CORE_IMAGE)       -f docker/core.Dockerfile .
	$(DOCKER) build -t $(TRAINER_IMAGE)    -f docker/trainer.Dockerfile .
	$(DOCKER) build -t $(VISUALIZER_IMAGE) -f docker/visualizer.Dockerfile .

docker-push: docker-build ## Push all Docker images
	$(DOCKER) push $(CORE_IMAGE)
	$(DOCKER) push $(TRAINER_IMAGE)
	$(DOCKER) push $(VISUALIZER_IMAGE)

# ══════════════════════════════════════════════════════════════
#  Clean
# ══════════════════════════════════════════════════════════════
.PHONY: clean clean-all

clean: ## Remove build artifacts
	rm -rf build/

clean-all: clean ## Remove build + Docker volumes
	$(DOCKER) compose -f $(COMPOSE_FILE) down -v --remove-orphans 2>/dev/null || true

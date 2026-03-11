SHELL := /bin/bash

CONAN ?= conan
CMAKE ?= cmake
CTEST ?= ctest
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
DOCKER ?= docker

BUILD_DIR_DEBUG := build/debug
BUILD_DIR_RELEASE := build/release
COMPOSE_FILE := docker-compose.yml

SRC_DIRS := src tests
FORMAT_FILES := $(shell find $(SRC_DIRS) -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \))

.PHONY: lint fmt-check tidy test build-debug build-release configure-debug configure-release up down logs

lint: fmt-check tidy

fmt-check:
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No C/C++ files found for formatting."; \
	else \
		$(CLANG_FORMAT) --dry-run --Werror $(FORMAT_FILES); \
	fi

tidy: configure-debug
	@if [ -z "$(FORMAT_FILES)" ]; then \
		echo "No C/C++ files found for clang-tidy."; \
	else \
		$(CLANG_TIDY) -p $(BUILD_DIR_DEBUG) $(FORMAT_FILES); \
	fi

test: build-debug
	$(CTEST) --preset debug

configure-debug:
	$(CONAN) install . --output-folder=$(BUILD_DIR_DEBUG) --build=missing
	$(CMAKE) --preset debug

configure-release:
	$(CONAN) install . --output-folder=$(BUILD_DIR_RELEASE) --build=missing
	$(CMAKE) --preset release

build-debug: configure-debug
	$(CMAKE) --build --preset debug

build-release: configure-release
	$(CMAKE) --build --preset release

up:
	$(DOCKER) compose -f $(COMPOSE_FILE) up -d

down:
	$(DOCKER) compose -f $(COMPOSE_FILE) down

logs:
	$(DOCKER) compose -f $(COMPOSE_FILE) logs -f --tail=200

# ADR-0009: Test Framework and CTest Discovery

Status: Accepted  
Date: 2026-03-07

## Context

The MVP needs consistent and extensible C++ tests that:
- are easy to read and maintain;
- integrate into CI through standard `ctest`;
- provide detailed reporting for each test case.

## Decision

1. C++ unit tests in the project are written with GoogleTest (`TEST`, `ASSERT_*`, `EXPECT_*`).
2. Manual test runners with a custom `main` and return-code checks are not used for new tests.
3. Test binaries are registered in CTest through `gtest_discover_tests(...)` so each case appears as a separate test.
4. The base test entrypoint remains `ctest` locally and in CI.

## Consequences

- Tests become uniform and scale more easily to the next MVP steps.
- Failure diagnostics improve through per-case reporting in `ctest`.
- CI gets a stable and predictable test loop without hand-written runners.

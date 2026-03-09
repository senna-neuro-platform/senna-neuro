# ADR-0001: Versioning and Changelog Policy

Status: Accepted  
Date: 2026-03-07

## Context

Project versions and the change log must be maintained consistently so that:
- the version clearly shows the MVP step and the iteration inside that step;
- increment rules are formal and verifiable;
- the changelog structure stays consistent across commits.

## Decision

1. Version format: `A.B.C-dev`.
2. `A` (major): always `0` until the first release.
3. `B` (minor): equals the number of the current MVP plan step.
4. `C` (patch/build): increases by `+1` for every change within the current step.
5. When moving to a new MVP step: `B = B + 1`, `C = 0`.
6. Single source of truth for the version: the `VERSION` file.
7. The changelog is maintained in `CHANGELOG.md`:
   - entries are grouped by `B` (minor), meaning one block per MVP step;
   - the block heading format is `## DD.MM.YYYY \`A.B.C-dev\`` (the current step version);
   - for changes inside the same `B`, only `C` changes, the version in the existing block heading is updated, and new bullet points are added to the top of that block;
   - a new block is created only when moving to a new `B`, and it is placed above the previous ones;
   - each block records only actual project changes; maintenance notes like "updated version/changelog" are not added;
   - order is always reverse chronological, with the newest items first.
8. Any code, build, runtime, or enforced workflow change is not considered complete until `VERSION`
   and the top corresponding block in `CHANGELOG.md` are updated in the same change set.

## Consequences

- Every change inside a step requires updating `VERSION` and the top block in `CHANGELOG.md` for the corresponding `B`.
- The current MVP step number is always read directly from `B`.
- Change history is preserved in reverse chronological order.

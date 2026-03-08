# ADR-0012: Training Target and Robustness Gates

- Status: Accepted
- Date: 2026-03-07

## Context

Requires moving from smoke training to a full MNIST run with a target accuracy and explicit robustness checks.

## Decision

1. The Python entrypoint `python/train.py` performs epoch training, evaluation on the test set, and writes JSONL metrics.
2. A checkpoint is stored for every epoch in `data/artifacts/outbox/epoch_XXXXXXXXX.h5`.
3. The target training criterion is fixed as `target_accuracy = 0.85` with early stop once it is reached.
4. After training, robustness checks are executed on the saved state:
   - `remove_neurons(0.1)` with allowed degradation `< 5%`;
   - `inject_noise(0.3)` with allowed degradation `< 10%`.
5. Diagnostic heuristics (`silent`, `epileptic`, `dominance`) are emitted into the log from runtime metrics.

## Consequences

- DoD thresholds become reproducible and script-verifiable.
- Experiment artifacts and metrics are ready for later analysis and background upload.

# ADR-0012: Training Target and Robustness Gates

- Status: Accepted
- Date: 2026-03-07

## Context

Шаг 15 требует перейти от smoke-training к полноценному прогону MNIST с целевой точностью и проверками робастности.

## Decision

1. Python entrypoint `python/train.py` выполняет epoch-training, eval на тестовом наборе и пишет JSONL-метрики.
2. Для каждой эпохи сохраняется checkpoint в формате `data/artifacts/outbox/epoch_XXXXXXXXX.h5`.
3. Целевой критерий обучения фиксируется как `target_accuracy = 0.85` (early-stop по достижению).
4. После обучения выполняются робастность-проверки на сохраненном состоянии:
   - `remove_neurons(0.1)` с допустимой деградацией `< 5%`;
   - `inject_noise(0.3)` с допустимой деградацией `< 10%`.
5. Диагностические эвристики шага 15 (silent/epileptic/dominance) выводятся в лог по runtime-метрикам.

## Consequences

- Пороговые критерии DoD шага 15 становятся воспроизводимыми и проверяемыми скриптом.
- Экспериментальные артефакты и метрики готовы для последующего анализа и фоновой выгрузки.

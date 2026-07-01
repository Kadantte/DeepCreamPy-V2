# TASK-007: Training CLI And Checkpoint Resume

## Agent

Claude Code Agent G

## Depends On

- `TASK-003-ARRAY-MASK-SEMANTICS.MD`
- `TASK-004-DATASET-CORE.MD`
- `TASK-005-MASK-GENERATORS.MD`
- `TASK-006-MOSAIC-SIMULATION.MD`

## Goal

Create a real training command for bar and mosaic models.

## Owned Files

You may create:

- `lib-cream-py/src/tools/train_model.py`
- `lib-cream-py/src/training/__init__.py`
- `lib-cream-py/src/training/config.py`
- `lib-cream-py/src/training/loop.py`
- `lib-cream-py/src/training/samples.py`
- `tests/test_training_smoke.py`
- `tests/test_training_resume.py`

You may edit:

- `lib-cream-py/src/model/model.py`
- `lib-cream-py/src/__init__.py`

## Required CLI

```bash
PYTHONPATH=lib-cream-py/src python -m tools.train_model \
  --kind bar \
  --image-dir ./data/clean \
  --work-dir ./runs/bar-test \
  --export ./models/bar.keras \
  --batch-size 1 \
  --steps 10 \
  --save-every 5
```

And:

```bash
PYTHONPATH=lib-cream-py/src python -m tools.train_model \
  --kind mosaic \
  --image-dir ./data/clean \
  --work-dir ./runs/mosaic-test \
  --export ./models/mosaic.keras
```

## Required Features

- `--kind`: `bar` or `mosaic`
- `--image-dir`
- `--work-dir`
- `--export`
- `--batch-size`
- `--steps`
- `--seed`
- `--resume`
- `--save-every`
- `--sample-every`
- `--learning-rate-g`
- `--learning-rate-d`

## Strict Requirements

1. No long default training run. Defaults must be safe.
2. Must support CPU-only smoke run.
3. Must save config to `work-dir/config.json`.
4. Must save checkpoints under `work-dir/checkpoints`.
5. Must save sample images under `work-dir/samples`.
6. Must export `.keras` only when requested or at final step.
7. Must checkpoint generator, discriminator, both optimizers, and global step.
8. Must be resumable.
9. Must fail clearly if image directory is empty.
10. Must not write into `models/` unless user explicitly passes `--export models/...`.

## Acceptance Tests

Smoke run with synthetic fixtures:

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_training_smoke.py
```

Resume test:

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_training_resume.py
```

Manual tiny run:

```bash
PYTHONPATH=lib-cream-py/src python -m tools.train_model \
  --kind bar \
  --image-dir /tmp/dcp_synthetic_clean \
  --work-dir /tmp/dcp_train_bar \
  --export /tmp/dcp_train_bar.keras \
  --batch-size 1 \
  --steps 2
```

## Done Means

- The repo has a real starting point for training new models.
- Long training is still optional/manual, but tiny training is automated and tested.


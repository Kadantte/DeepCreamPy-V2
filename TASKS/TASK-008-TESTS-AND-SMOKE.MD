# TASK-008: Test Suite And Smoke Coverage

## Agent

Claude Code Agent H

## Depends On

- Most earlier implementation tasks.

## Goal

Create a regression test suite that proves checkpoint migration, masks, dataset generation, inference, and training smoke behavior work.

## Owned Files

You may create/edit:

- `tests/`
- `pytest.ini`
- `lib-cream-py/pyproject.toml` test config section only
- `.github/workflows/test.yml` only if `TASK-009` is not active

## Required Test Groups

### Unit Tests

- public imports
- mask normalization
- ArrayMask behavior
- connected region finding
- bounding box expansion
- image preprocessing
- training mask generators
- mosaic simulation

### Integration Tests

- checkpoint inspection on a tiny fake checkpoint if feasible
- Keras custom layer save/load
- `InpaintNN(create_model=True)` builds variables
- `.keras` save/load/predict smoke
- one training step on synthetic data

### Optional Slow Tests

These must be skipped by default unless an env var is set:

- read real `models/bar/Train_775000`
- migrate real checkpoint to `/tmp`
- run real mermaid inference smoke

Use:

```text
DCP_RUN_SLOW_TESTS=1
```

## Strict Requirements

1. Tests must not require downloaded model files by default.
2. Tests must not require GPU.
3. Tests must not require internet.
4. Tests must not use adult images.
5. Slow tests must be opt-in.
6. Generated files must go to temp dirs.
7. No tests should write into `models/` by default.

## Acceptance Commands

```bash
cd /mnt/c/Users/Usui/Desktop/Projects/DeepCreamPy-V2
source /home/usui/.venvs/deepcreampy-v2/bin/activate
PYTHONPATH=lib-cream-py/src pytest
```

Optional:

```bash
DCP_RUN_SLOW_TESTS=1 PYTHONPATH=lib-cream-py/src pytest
```

## Done Means

- Future changes can be checked without guessing whether inference/training broke.
- Expensive real-model tests are available but not mandatory.


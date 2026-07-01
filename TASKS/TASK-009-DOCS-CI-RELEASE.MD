# TASK-009: Docs, CI, And Release Integration

## Agent

Claude Code Agent I

## Depends On

- `TASK-001`
- `TASK-007`
- `TASK-008`

## Goal

Update docs and CI so migration, dataset generation, training smoke, and release expectations are clear and repeatable.

## Owned Files

You may edit:

- `README.md`
- `docs/INSTALLATION.md`
- `docs/USAGE.md`
- `TRAINING_REVERSE_ENGINEERING.md`
- `ROADMAP.MD`
- `.github/workflows/*.yml`
- `Dockerfile`

You may create:

- `docs/TRAINING.md`
- `docs/MODEL_MIGRATION.md`
- `.github/workflows/test.yml`

## Required Docs

Create/update docs for:

- Setting up WSL/venv safely.
- Installing TensorFlow/Keras.
- Inspecting TF1 checkpoints.
- Migrating `Train_775000` and `Train_290000`.
- Running inference with `.keras`.
- Generating a training dataset.
- Running tiny training smoke.
- Why full training is expensive and not automatic.

## CI Requirements

Add CI that runs:

- Python import/unit tests.
- Mask/data tests.
- Keras save/load smoke if feasible.
- Frontend typecheck/build if dependencies are available.

Do not make CI require:

- real model checkpoints
- GPU
- adult content
- long training

## Docker Requirements

If editing Dockerfile:

1. Fix stale references like `requirements-cpu.txt` if missing.
2. Fix stale entrypoint like `decensor.py` if missing.
3. Keep image build CPU-safe.
4. Do not bake downloaded models into the image.

## Strict Requirements

1. Docs must distinguish TF1 checkpoints from `.keras` files.
2. Docs must explain that `models/*.keras` is ignored.
3. Docs must include exact commands.
4. Docs must not claim full training quality is solved until tiny-overfit and real training are verified.
5. CI must be practical for pull requests.

## Acceptance Tests

Manual doc command check:

```bash
grep -R "migrate_checkpoint" docs README.md TRAINING_REVERSE_ENGINEERING.md
grep -R "train_model" docs README.md TRAINING_REVERSE_ENGINEERING.md
```

CI config check:

```bash
git diff -- .github/workflows
```

## Done Means

- A new contributor can migrate models and run smoke training from docs.
- CI protects the core training/inference path.


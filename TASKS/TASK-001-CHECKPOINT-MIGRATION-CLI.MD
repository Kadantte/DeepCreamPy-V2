# TASK-001: Checkpoint Migration CLI

## Agent

Claude Code Agent A

## Goal

Create a reliable command-line tool to convert old TensorFlow 1 checkpoints into app-ready `.keras` models.

## Why

The downloaded models are checkpoint prefixes:

- `models/bar/Train_775000`
- `models/mosaic/Train_290000`

The current app expects:

- `models/bar.keras`
- `models/mosaic.keras`

Manual one-line Python migration works, but the repo needs a repeatable tool.

## Owned Files

You may create:

- `lib-cream-py/src/tools/__init__.py`
- `lib-cream-py/src/tools/migrate_checkpoint.py`
- `lib-cream-py/src/tools/inspect_checkpoint.py`
- `tests/test_checkpoint_tools.py`

You may edit only if needed:

- `lib-cream-py/src/model/model.py`
- `TRAINING_REVERSE_ENGINEERING.md`

## Do Not Touch

- `models/bar/*`
- `models/mosaic/*`
- `models/*.keras`
- frontend files
- server queue files

## Required CLI Behavior

Add:

```bash
PYTHONPATH=lib-cream-py/src python -m tools.migrate_checkpoint \
  --checkpoint models/bar/Train_775000 \
  --out models/bar.keras \
  --kind bar
```

And:

```bash
PYTHONPATH=lib-cream-py/src python -m tools.inspect_checkpoint \
  --checkpoint models/bar/Train_775000
```

## Required Options

`migrate_checkpoint.py`:

- `--checkpoint`: required TF1 checkpoint prefix.
- `--out`: required `.keras` output path.
- `--kind`: choices `bar`, `mosaic`, `unknown`; used only for metadata/reporting.
- `--overwrite`: allow replacing existing output.
- `--dry-run`: validate mapping without writing.

`inspect_checkpoint.py`:

- `--checkpoint`: required checkpoint prefix.
- `--json`: optional JSON output.
- `--show-vars`: print variable list.

## Strict Requirements

1. Fail if `.index` or `.data-00000-of-00001` is missing.
2. Fail if output exists and `--overwrite` is not passed.
3. Validate all generator variable mappings.
4. Validate all discriminator variable mappings.
5. Print count of assigned variables.
6. Print ignored optimizer variable count.
7. Save `.keras` only after all assignment checks pass.
8. Run a one-sample prediction after saving and reloading.
9. Do not import server/frontend code.
10. Do not make migration depend on current working directory.

## Acceptance Tests

Run in WSL:

```bash
cd /mnt/c/Users/Usui/Desktop/Projects/DeepCreamPy-V2
source /home/usui/.venvs/deepcreampy-v2/bin/activate
PYTHONPATH=lib-cream-py/src python -m tools.inspect_checkpoint --checkpoint models/bar/Train_775000
PYTHONPATH=lib-cream-py/src python -m tools.migrate_checkpoint --checkpoint models/bar/Train_775000 --out /tmp/bar.keras --kind bar --overwrite
PYTHONPATH=lib-cream-py/src python -m tools.migrate_checkpoint --checkpoint models/mosaic/Train_290000 --out /tmp/mosaic.keras --kind mosaic --overwrite
```

Expected:

- Both `/tmp/*.keras` files are created.
- Both reload successfully.
- Prediction smoke test passes.

## Done Means

- Migration no longer requires hand-written Python one-liners.
- Bar and mosaic migrations can run in the same Python process.
- Tool emits enough metadata to debug future checkpoint mismatches.


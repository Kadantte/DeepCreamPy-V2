# TASK-002: Public Package API And Import Cleanup

## Agent

Claude Code Agent B

## Goal

Make `lib-cream-py` usable as a real Python library without fragile cwd-dependent imports.

## Why

The current library has an empty `__init__.py` and many imports like:

```python
from logger import Logger
from mask import Mask
from model.model import InpaintNN
```

These work only under certain `PYTHONPATH`/cwd setups.

## Owned Files

You may edit:

- `lib-cream-py/src/__init__.py`
- `lib-cream-py/src/lib_cream_py.py`
- `lib-cream-py/src/mask.py`
- `lib-cream-py/src/util.py`
- `lib-cream-py/src/model/*.py`
- `lib-cream-py/pyproject.toml`

You may create:

- `tests/test_public_api.py`

## Do Not Touch

- server behavior
- frontend files
- dataset modules from other tasks
- checkpoint files

## Required Public API

`from lib_cream_py import ...` must expose:

- `InpaintNN`
- `InpaintModel`
- `Mask`
- `ColorMask`
- `RawMask`
- `RawMask2`
- `Logger`
- `decensor_image`
- `decensor_image_variations`
- `apply_variant`
- `image_to_array`
- `find_regions`
- `expand_bounding`

## Strict Requirements

1. Use package-relative imports where possible.
2. Preserve backwards compatibility for current server imports if feasible.
3. Do not rename the distribution package without updating `pyproject.toml`.
4. Do not break `PYTHONPATH=lib-cream-py/src python -c "from lib_cream_py import InpaintNN"`.
5. Do not require TensorFlow import for mask-only utilities if avoidable.
6. Add tests that import the public API.

## Acceptance Tests

```bash
cd /mnt/c/Users/Usui/Desktop/Projects/DeepCreamPy-V2
source /home/usui/.venvs/deepcreampy-v2/bin/activate
PYTHONPATH=lib-cream-py/src python - <<'PY'
from lib_cream_py import InpaintNN, ColorMask, RawMask, decensor_image
print("public api ok")
PY
```

If `pytest` is available:

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_public_api.py
```

## Done Means

- The library has a clear import surface.
- Future tools can import library code without relying on cwd accidents.


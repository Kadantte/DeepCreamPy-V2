# TASK-006: Mosaic Censor Simulation

## Agent

Claude Code Agent F

## Depends On

- `TASK-004-DATASET-CORE.MD`
- `TASK-005-MASK-GENERATORS.MD`

## Goal

Create mosaic censor simulation for generating `X` inputs for mosaic model training.

## Owned Files

You may create:

- `lib-cream-py/src/data/mosaic.py`
- `tests/test_data_mosaic.py`

You may edit:

- `lib-cream-py/src/data/__init__.py`

## Required Functions

```python
apply_pixelation(image_array, hole_mask, block_size)
random_mosaic_block_size(rng, min_block, max_block)
make_mosaic_censored_input(clean_array, hole_mask, rng, min_block=4, max_block=16)
```

## Input / Output Contract

Input:

- `clean_array`: float32 `(H, W, 3)` in `[-1, 1]`
- `hole_mask`: bool `(H, W)`, True means censored/reconstruct region

Output:

- `x`: float32 `(H, W, 3)` in `[-1, 1]`

Behavior:

- Pixels outside `hole_mask` must remain unchanged.
- Pixels inside `hole_mask` become pixelated/mosaic.

## Strict Requirements

1. Do not alter pixels outside the hole mask.
2. Keep output range in `[-1, 1]`.
3. Support deterministic output with seeded RNG.
4. No TensorFlow dependency.
5. Tests must use synthetic arrays.
6. Block size must be configurable.
7. Handle masks touching image borders.

## Acceptance Tests

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_data_mosaic.py
```

Manual smoke:

```python
import numpy as np
from data.mosaic import make_mosaic_censored_input

rng = np.random.default_rng(123)
clean = rng.uniform(-1, 1, (256, 256, 3)).astype("float32")
mask = np.zeros((256, 256), dtype=bool)
mask[80:160, 80:160] = True
x = make_mosaic_censored_input(clean, mask, rng)
assert np.allclose(x[~mask], clean[~mask])
assert not np.allclose(x[mask], clean[mask])
```

## Done Means

- Mosaic training samples can be generated from clean images.
- The mosaic model no longer depends on hand-prepared mosaic pairs.


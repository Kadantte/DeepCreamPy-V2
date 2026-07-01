# TASK-005: Training Mask Generators

## Agent

Claude Code Agent E

## Depends On

- `TASK-004-DATASET-CORE.MD`
- Ideally `TASK-003-ARRAY-MASK-SEMANTICS.MD`

## Goal

Create mask generators used for training bar and general inpainting models.

## Owned Files

You may create:

- `lib-cream-py/src/data/masks.py`
- `tests/test_data_masks.py`

You may edit:

- `lib-cream-py/src/data/__init__.py`

## Required Mask Types

Implement:

- rectangular/bar masks
- random square masks
- freeform brush-stroke masks
- combined masks
- deterministic masks via seed

## Required Functions

```python
random_rectangle_mask(height, width, rng, min_size, max_size)
random_bar_mask(height, width, rng, min_bars, max_bars)
random_freeform_mask(height, width, rng, strokes, brush_width)
combine_masks(*masks)
mask_to_known_mask(hole_mask)
```

## Mask Semantics

Use:

```text
hole_mask: bool ndarray, True = reconstruct this area
known_mask: float32 ndarray, 1 = visible/known, 0 = hole
```

## Strict Requirements

1. Every generator must return `(H, W)` boolean `hole_mask`.
2. Every generator must accept a NumPy random generator.
3. Same seed must produce same mask.
4. Masks must never be empty unless explicitly requested.
5. Masks must never exceed image bounds.
6. Add tests for determinism.
7. Add tests for coverage ratio bounds.
8. Do not import TensorFlow.

## Acceptance Tests

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_data_masks.py
```

Manual smoke:

```python
import numpy as np
from data.masks import random_bar_mask

rng = np.random.default_rng(123)
mask = random_bar_mask(256, 256, rng, 1, 4)
assert mask.dtype == bool
assert mask.any()
```

## Done Means

- Training can synthesize masks without manually edited green images.
- Mask output can be plugged into dataset pipeline and future UI/detector APIs.


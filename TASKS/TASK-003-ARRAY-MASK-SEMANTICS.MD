# TASK-003: ArrayMask And Mask Semantics

## Agent

Claude Code Agent D

## Depends On

- Preferably `TASK-002` public API cleanup.

## Goal

Make masks first-class ndarray inputs, not color-derived by default.

## Why

The roadmap goal says:

> mask isn't based on color anymore

The current `ColorMask` works by exact RGB equality. That is fragile and prevents detector software from passing an ndarray mask directly.

## Owned Files

You may edit:

- `lib-cream-py/src/mask.py`
- `lib-cream-py/src/util.py`
- `lib-cream-py/src/lib_cream_py.py`
- `lib-cream-py/src/__init__.py`

You may create:

- `tests/test_masks.py`
- `tests/test_decensor_array_mask.py`

## Required Concepts

Internal standard:

```text
hole_mask: bool ndarray, True = area to reconstruct
known_mask: float tensor, 1 = visible/known, 0 = hidden/hole
```

Existing inference `Mask.find_mask_simple()` currently returns `1 = selected/masked region`.

Do not silently mix these conventions. Add conversion helpers with clear names.

## Required Classes / Functions

Add:

- `ArrayMask`
- `normalize_hole_mask(mask, height=None, width=None) -> np.ndarray`
- `hole_mask_to_known_mask(mask) -> np.ndarray`
- `known_mask_to_hole_mask(mask) -> np.ndarray`

`ArrayMask` must accept:

- `(H, W)` bool
- `(H, W)` uint8 `0/1`
- `(H, W)` uint8 `0/255`
- `(H, W, 1)`
- `(H, W, 3)` where channels agree
- float masks in `0..1`

## Required Inference Change

Add an ndarray-friendly path such as:

```python
decensor_image(model, mask, ori, colored=None, is_mosaic=False, logger=...)
```

`colored` must become optional when a mask array is supplied.

## Strict Requirements

1. Keep `ColorMask` working for legacy green mask workflows.
2. Keep `RawMask` working for mask image files.
3. Validate mask/image dimensions.
4. Fail clearly for ambiguous 3-channel masks whose channels differ.
5. Do not use exact RGB matching in `ArrayMask`.
6. Tests must cover all accepted shapes/dtypes.
7. Tests must cover bad shapes and mismatched dimensions.

## Acceptance Tests

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_masks.py
```

Manual smoke:

```python
import numpy as np
from PIL import Image
from lib_cream_py import ArrayMask

img = Image.new("RGB", (256, 256))
mask = np.zeros((256, 256), dtype=bool)
mask[64:128, 64:128] = True
m = ArrayMask(mask)
assert m.find_mask_simple().sum() == 4096
```

## Done Means

- Detector software can pass `np.ndarray` masks directly.
- Color masks are compatibility adapters, not the core abstraction.


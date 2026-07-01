# TASK-004: Dataset Core Pipeline

## Agent

Claude Code Agent C

## Goal

Create the base dataset/preprocessing pipeline that turns clean images into normalized `Y` tensors and leaves hooks for mask/censor generators.

## Why

Training cannot proceed without batches shaped like:

```text
X: censored image, float32, shape [B, 256, 256, 3], range [-1, 1]
Y: clean target, float32, shape [B, 256, 256, 3], range [-1, 1]
MASK: known mask, float32, shape [B, 256, 256, 3], 1 = visible, 0 = hole
```

This task owns image loading and batch construction scaffolding only. Mask shape generation belongs to `TASK-005`; mosaic simulation belongs to `TASK-006`.

## Owned Files

You may create:

- `lib-cream-py/src/data/__init__.py`
- `lib-cream-py/src/data/images.py`
- `lib-cream-py/src/data/pipeline.py`
- `lib-cream-py/src/data/debug_export.py`
- `tests/test_data_images.py`
- `tests/test_data_pipeline.py`

You may edit:

- `lib-cream-py/src/__init__.py`

## Required Functions

In `images.py`:

- `iter_image_paths(root, extensions=...)`
- `load_rgb_image(path)`
- `center_crop_square(image)`
- `resize_image(image, size=(256, 256))`
- `image_to_training_array(image)`
- `training_array_to_image(array)`

In `pipeline.py`:

- `TrainingSample`
- `build_sample(clean_image, hole_mask, censor_fn=None)`
- `batch_samples(samples)`
- `make_dataset_from_paths(paths, mask_fn, censor_fn, batch_size, shuffle=False)`

## Strict Requirements

1. No adult fixture images.
2. Tests must use generated synthetic images.
3. No TensorFlow dependency required for basic image preprocessing tests.
4. Normalize images to `[-1, 1]`.
5. Provide inverse conversion back to PIL for debugging.
6. Always output RGB, never RGBA/P/indexed.
7. Do not implement random mask generation here; accept `mask_fn`.
8. Do not implement mosaic logic here; accept `censor_fn`.

## Acceptance Tests

```bash
PYTHONPATH=lib-cream-py/src pytest tests/test_data_images.py tests/test_data_pipeline.py
```

Manual smoke:

```python
from pathlib import Path
from data.images import iter_image_paths, load_rgb_image, image_to_training_array

paths = list(iter_image_paths(Path("decensor_input")))
img = load_rgb_image(paths[0])
arr = image_to_training_array(img)
assert arr.shape == (256, 256, 3)
assert arr.min() >= -1 and arr.max() <= 1
```

## Done Means

- Clean images can become normalized `Y`.
- A mask/censor function can plug in later to produce `X` and `MASK`.


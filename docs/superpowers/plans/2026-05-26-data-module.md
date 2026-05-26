# D4RT Data Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `data/` package so `train.py --dataset pointodyssey` runs end-to-end. Other dataset names ship as stubs.

**Architecture:** Layered. `data/base.py` holds common helpers; `data/pointodyssey.py` is the only fully-implemented dataset; `data/query_sampling.py` + `data/targets.py` are stand-alone helpers; `data/stubs.py` exposes placeholder classes for Video/Kubric/Sintel/ScanNet. Tests live in a single file `tests/test_data.py` and use a synthetic on-disk fixture so they need no real PointOdyssey data.

**Tech Stack:** Python 3, PyTorch, torchvision, NumPy, Pillow, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-25-data-module-design.md` (read it first).

---

## File Structure

Files this plan creates / modifies:

| Path | Status | Responsibility |
|---|---|---|
| `data/__init__.py` | create | Re-export `PointOdysseyDataset`, `VideoDataset`, `KubricDataset`, `SintelDataset`, `ScanNetDataset`, `collate_fn` |
| `data/augmentations.py` | create | `AugmentationConfig`, `VideoAugmentation`, `TemporalSubsampling` |
| `data/stubs.py` | create | Four stub dataset classes that raise `NotImplementedError` |
| `data/collate.py` | create | `collate_fn` for nested dicts |
| `data/base.py` | create | `BaseVideoDataset` with resize/normalize/aspect-ratio helpers |
| `data/query_sampling.py` | create | `sample_queries()` — mixed-task per-sample query generation |
| `data/targets.py` | create | `build_targets()` — produces loss-ready targets dict |
| `data/pointodyssey.py` | create | `PointOdysseyDataset` |
| `tests/test_data.py` | create | All tests + synthetic fixture builder |
| `train.py` | modify | Add `'pointodyssey'` to `--dataset` choices and a `create_dataloader` branch |

---

## Task 1: Augmentations

**Files:**
- Create: `data/augmentations.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create `data/` package init (empty for now)**

Create `data/__init__.py` with content:

```python
"""D4RT data loading package."""
```

- [ ] **Step 2: Write the failing tests for augmentations**

Create `tests/test_data.py` with content:

```python
"""Tests for the data package."""

import numpy as np
import pytest
import torch

from data.augmentations import AugmentationConfig, VideoAugmentation, TemporalSubsampling


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def test_augmentation_config_defaults():
    cfg = AugmentationConfig()
    assert cfg.brightness == 0.2
    assert cfg.contrast == 0.2
    assert cfg.saturation == 0.2
    assert cfg.hue == 0.05
    assert cfg.blur_prob == 0.1
    assert cfg.temporal_stride_min == 1
    assert cfg.temporal_stride_max == 4


def test_video_augmentation_preserves_shape_and_dtype():
    aug = VideoAugmentation(AugmentationConfig())
    video = np.random.rand(8, 32, 32, 3).astype(np.float32)
    out = aug(video)
    assert out.shape == video.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_video_augmentation_is_deterministic_within_clip():
    """All T frames must share the SAME color transform (no flicker)."""
    aug = VideoAugmentation(AugmentationConfig(brightness=0.5, contrast=0.0,
                                                saturation=0.0, hue=0.0,
                                                blur_prob=0.0))
    # Two identical frames -> after aug they should still be identical
    frame = np.full((32, 32, 3), 0.5, dtype=np.float32)
    video = np.stack([frame, frame], axis=0)
    out = aug(video)
    np.testing.assert_allclose(out[0], out[1], atol=1e-5)


def test_temporal_subsampling_range():
    cfg = AugmentationConfig(temporal_stride_min=1, temporal_stride_max=4)
    ts = TemporalSubsampling(cfg)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = ts.sample_stride(rng)
        assert 1 <= s <= 4
        assert isinstance(s, int)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /jfs/jing.feng/codebases/d4rt-pytorch
pytest tests/test_data.py -v
```

Expected: `ImportError` / `ModuleNotFoundError` on `data.augmentations`.

- [ ] **Step 4: Implement `data/augmentations.py`**

Create `data/augmentations.py`:

```python
"""Video augmentation utilities for D4RT training.

Implements photometric augmentation that is consistent across all frames of a
clip (so the tracking signal is preserved) plus a stateless helper for
temporal subsampling.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import functional as TF


@dataclass
class AugmentationConfig:
    """Configuration for video augmentation."""
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    blur_prob: float = 0.1
    blur_sigma_max: float = 1.5
    temporal_stride_min: int = 1
    temporal_stride_max: int = 4


class VideoAugmentation:
    """Per-clip photometric augmentation.

    All frames in a clip receive the SAME color transform parameters
    (re-sampled per __call__). This preserves temporal coherence — per-frame
    resampling would cause flicker and destroy the tracking signal.
    """

    def __init__(self, cfg: Optional[AugmentationConfig] = None):
        self.cfg = cfg or AugmentationConfig()
        self._rng = np.random.default_rng()

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Apply photometric augmentation to a video clip.

        Args:
            video: (T, H, W, 3) float32 in [0, 1].

        Returns:
            (T, H, W, 3) float32 in [0, 1].
        """
        if video.dtype != np.float32:
            video = video.astype(np.float32)

        # (T, H, W, C) -> (T, C, H, W) torch tensor for torchvision functional
        t = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()

        c = self.cfg
        # Sample once per clip
        if c.brightness > 0:
            b = float(self._rng.uniform(max(0.0, 1.0 - c.brightness), 1.0 + c.brightness))
            t = TF.adjust_brightness(t, b)
        if c.contrast > 0:
            ct = float(self._rng.uniform(max(0.0, 1.0 - c.contrast), 1.0 + c.contrast))
            t = TF.adjust_contrast(t, ct)
        if c.saturation > 0:
            s = float(self._rng.uniform(max(0.0, 1.0 - c.saturation), 1.0 + c.saturation))
            t = TF.adjust_saturation(t, s)
        if c.hue > 0:
            h = float(self._rng.uniform(-c.hue, c.hue))
            t = TF.adjust_hue(t, h)
        if c.blur_prob > 0 and self._rng.random() < c.blur_prob:
            sigma = float(self._rng.uniform(0.1, c.blur_sigma_max))
            kernel = max(3, int(2 * round(sigma * 3) + 1))
            if kernel % 2 == 0:
                kernel += 1
            t = TF.gaussian_blur(t, kernel_size=[kernel, kernel], sigma=[sigma, sigma])

        t = t.clamp(0.0, 1.0)
        return t.permute(0, 2, 3, 1).contiguous().numpy()


class TemporalSubsampling:
    """Stateless helper for sampling a temporal stride per clip."""

    def __init__(self, cfg: Optional[AugmentationConfig] = None):
        self.cfg = cfg or AugmentationConfig()

    def sample_stride(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.cfg.temporal_stride_min,
                                self.cfg.temporal_stride_max + 1))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/__init__.py data/augmentations.py tests/test_data.py
git commit -m "feat(data): augmentation utilities (photometric + temporal)"
```

---

## Task 2: Stubs and package exports

**Files:**
- Create: `data/stubs.py`
- Modify: `data/__init__.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append failing stub tests to `tests/test_data.py`**

Add at the end of `tests/test_data.py`:

```python
# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

import pytest as _pytest  # noqa: E402

from data import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset  # noqa: E402


@_pytest.mark.parametrize("cls", [VideoDataset, KubricDataset, SintelDataset, ScanNetDataset])
def test_stub_dataset_raises(cls):
    with _pytest.raises(NotImplementedError) as exc:
        cls("/tmp/does-not-matter", split="train", num_frames=4,
            img_size=64, num_queries=32, transform=None)
    assert "pointodyssey" in str(exc.value).lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` from `from data import VideoDataset, ...`.

- [ ] **Step 3: Create `data/stubs.py`**

```python
"""Placeholder dataset classes.

These exist to satisfy `train.py`'s imports. They raise NotImplementedError on
construction with a clear pointer to PointOdyssey, which is the fully
implemented dataset for now.
"""


class _StubDataset:
    def __init__(self, data_root, split='train', num_frames=48, img_size=256,
                 num_queries=2048, transform=None, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} is not implemented yet. "
            f"Use --dataset pointodyssey. "
            f"See docs/superpowers/specs/2026-05-25-data-module-design.md"
        )


class VideoDataset(_StubDataset):
    pass


class KubricDataset(_StubDataset):
    pass


class SintelDataset(_StubDataset):
    pass


class ScanNetDataset(_StubDataset):
    pass
```

- [ ] **Step 4: Update `data/__init__.py` to re-export stubs**

Overwrite `data/__init__.py`:

```python
"""D4RT data loading package."""

from data.stubs import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset

__all__ = [
    "VideoDataset",
    "KubricDataset",
    "SintelDataset",
    "ScanNetDataset",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 8 tests pass (4 from Task 1 + 4 parametrized stub tests).

- [ ] **Step 6: Commit**

```bash
git add data/stubs.py data/__init__.py tests/test_data.py
git commit -m "feat(data): add stub datasets so train.py imports resolve"
```

---

## Task 3: Collate function

**Files:**
- Create: `data/collate.py`
- Modify: `data/__init__.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append failing collate test to `tests/test_data.py`**

```python
# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

from data import collate_fn  # noqa: E402


def _fake_sample(n_queries=4):
    return {
        "video": torch.zeros(2, 8, 8, 3, dtype=torch.float32),
        "coords": torch.zeros(n_queries, 2, dtype=torch.float32),
        "t_src": torch.zeros(n_queries, dtype=torch.long),
        "t_tgt": torch.zeros(n_queries, dtype=torch.long),
        "t_cam": torch.zeros(n_queries, dtype=torch.long),
        "aspect_ratio": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "targets": {
            "pos_3d": torch.zeros(n_queries, 3, dtype=torch.float32),
            "mask_3d": torch.zeros(n_queries, dtype=torch.float32),
        },
    }


def test_collate_stacks_top_level_and_nested():
    batch = [_fake_sample(), _fake_sample()]
    out = collate_fn(batch)

    assert out["video"].shape == (2, 2, 8, 8, 3)
    assert out["coords"].shape == (2, 4, 2)
    assert out["t_src"].shape == (2, 4)
    assert out["aspect_ratio"].shape == (2, 2)
    assert isinstance(out["targets"], dict)
    assert out["targets"]["pos_3d"].shape == (2, 4, 3)
    assert out["targets"]["mask_3d"].shape == (2, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_collate_stacks_top_level_and_nested -v
```

Expected: `ImportError` on `from data import collate_fn`.

- [ ] **Step 3: Create `data/collate.py`**

```python
"""Custom collate function for D4RT batches.

Handles the nested `targets` dict produced by `PointOdysseyDataset.__getitem__`.
"""

import torch


def collate_fn(batch):
    """Stack a list of sample dicts into a batched dict.

    Each sample is a dict with top-level tensors and one nested `targets` dict
    of tensors. Top-level tensors are stacked along dim 0; nested target
    tensors are stacked the same way under the `targets` key.
    """
    out = {}
    for k in batch[0]:
        if k == "targets":
            out[k] = {
                tk: torch.stack([b["targets"][tk] for b in batch])
                for tk in batch[0]["targets"]
            }
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
```

- [ ] **Step 4: Re-export `collate_fn` from `data/__init__.py`**

Overwrite `data/__init__.py`:

```python
"""D4RT data loading package."""

from data.collate import collate_fn
from data.stubs import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset

__all__ = [
    "VideoDataset",
    "KubricDataset",
    "SintelDataset",
    "ScanNetDataset",
    "collate_fn",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/collate.py data/__init__.py tests/test_data.py
git commit -m "feat(data): collate_fn for nested targets dict"
```

---

## Task 4: Base dataset helpers

**Files:**
- Create: `data/base.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append failing base-helper tests to `tests/test_data.py`**

```python
# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------

from data.base import resize_frames_square, compute_aspect_ratio, to_float32_normalized  # noqa: E402


def test_resize_frames_square_changes_only_HW():
    frames = np.zeros((4, 100, 200, 3), dtype=np.uint8)
    out = resize_frames_square(frames, 64)
    assert out.shape == (4, 64, 64, 3)
    assert out.dtype == np.uint8


def test_compute_aspect_ratio_normalizes_by_max():
    # Landscape: H=480, W=640 -> max=640 -> (480/640, 640/640) = (0.75, 1.0)
    ar = compute_aspect_ratio(480, 640)
    assert ar.shape == (2,)
    np.testing.assert_allclose(ar, [0.75, 1.0], atol=1e-6)


def test_to_float32_normalized_uint8_input():
    arr = np.array([[0, 128, 255]], dtype=np.uint8)
    out = to_float32_normalized(arr)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 128 / 255, 1.0]], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` on `data.base`.

- [ ] **Step 3: Create `data/base.py`**

```python
"""Common helpers for video datasets.

These are free functions rather than methods on a base class because:
- They have no shared state.
- They're useful from any context (PointOdyssey, future Kubric/Sintel/ScanNet,
  evaluation code).
"""

import numpy as np
from PIL import Image


def resize_frames_square(frames: np.ndarray, size: int) -> np.ndarray:
    """Resize each frame to a square (size x size) using bilinear interpolation.

    Args:
        frames: (T, H, W, 3) uint8 array.
        size: Target side length.

    Returns:
        (T, size, size, 3) uint8 array.
    """
    T = frames.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for i in range(T):
        img = Image.fromarray(frames[i])
        img = img.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(img)
    return out


def compute_aspect_ratio(orig_h: int, orig_w: int) -> np.ndarray:
    """Return (h/max, w/max) so encoders can distinguish landscape vs portrait.

    The longer side becomes 1.0; the shorter side is the ratio.
    """
    m = max(orig_h, orig_w)
    return np.array([orig_h / m, orig_w / m], dtype=np.float32)


def to_float32_normalized(arr: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] to float32 [0, 1]. Floats pass through unchanged."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)
```

Class API note: the spec uses `BaseVideoDataset` as a "container of helpers", but functionally these are stateless free functions. We export them as functions; if a base class is needed later it can wrap these. Mention in commit message.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/base.py tests/test_data.py
git commit -m "feat(data): base helpers (resize, aspect ratio, normalize)

Implemented as free functions rather than a base class — they have no
shared state and are useful from any caller. A class wrapper can be
added later if a real common dataset interface emerges."
```

---

## Task 5: Synthetic test fixture

**Files:**
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append fixture builder and a smoke test for it**

Append to `tests/test_data.py`:

```python
# ---------------------------------------------------------------------------
# Synthetic PointOdyssey fixture
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from PIL import Image as _PILImage  # noqa: E402


def make_fake_pointodyssey(root: Path, *, num_sequences: int = 2,
                            num_frames: int = 32, img_h: int = 64, img_w: int = 64,
                            num_trajs: int = 40, split: str = "train") -> Path:
    """Build a tiny synthetic PointOdyssey-shaped dataset on disk.

    Layout matches the spec §4 expectation:
      {root}/{split}/seq_<i>/{rgbs,depths,normals}/...
      {root}/{split}/seq_<i>/{anno.npz, intrinsics.npy, extrinsics.npy}

    Returns:
        The split directory: {root}/{split}.
    """
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for s in range(num_sequences):
        seq_dir = split_dir / f"seq_{s:04d}"
        (seq_dir / "rgbs").mkdir(parents=True, exist_ok=True)
        (seq_dir / "depths").mkdir(parents=True, exist_ok=True)
        (seq_dir / "normals").mkdir(parents=True, exist_ok=True)

        # RGB frames as JPEGs
        for t in range(num_frames):
            arr = (rng.random((img_h, img_w, 3)) * 255).astype(np.uint8)
            _PILImage.fromarray(arr).save(seq_dir / "rgbs" / f"rgb_{t:05d}.jpg",
                                          quality=95)

        # Depth + normals as per-frame .npy
        for t in range(num_frames):
            depth = (rng.random((img_h, img_w)).astype(np.float32) * 5.0 + 0.5)
            np.save(seq_dir / "depths" / f"depth_{t:05d}.npy", depth)
            normal = rng.standard_normal((img_h, img_w, 3)).astype(np.float32)
            normal /= (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
            np.save(seq_dir / "normals" / f"normal_{t:05d}.npy", normal)

        # Trajectories: T x P x 2/3, visibilities T x P
        trajs_2d = (rng.random((num_frames, num_trajs, 2)).astype(np.float32)
                    * np.array([img_w - 1, img_h - 1], dtype=np.float32))
        trajs_3d = rng.standard_normal((num_frames, num_trajs, 3)).astype(np.float32)
        trajs_3d[..., 2] = np.abs(trajs_3d[..., 2]) + 0.5  # positive Z
        visibilities = (rng.random((num_frames, num_trajs)) > 0.1).astype(np.float32)
        np.savez(seq_dir / "anno.npz",
                 trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=visibilities)

        # Camera matrices
        K = np.array([[img_w * 0.8, 0, img_w / 2],
                      [0, img_h * 0.8, img_h / 2],
                      [0, 0, 1]], dtype=np.float32)
        intrinsics = np.broadcast_to(K, (num_frames, 3, 3)).copy()
        np.save(seq_dir / "intrinsics.npy", intrinsics)

        extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32),
                                      (num_frames, 4, 4)).copy()
        np.save(seq_dir / "extrinsics.npy", extrinsics)

    return split_dir


def test_fake_pointodyssey_layout(tmp_path):
    root = make_fake_pointodyssey(tmp_path, num_sequences=2, num_frames=8)
    assert (root / "seq_0000" / "rgbs" / "rgb_00000.jpg").exists()
    assert (root / "seq_0000" / "depths" / "depth_00007.npy").exists()
    assert (root / "seq_0001" / "anno.npz").exists()
    anno = np.load(root / "seq_0000" / "anno.npz")
    assert {"trajs_2d", "trajs_3d", "visibilities"} <= set(anno.files)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 13 tests pass. The fixture builder is exercised by `test_fake_pointodyssey_layout`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data.py
git commit -m "test(data): synthetic PointOdyssey fixture for offline testing"
```

---

## Task 6: Query sampling

**Files:**
- Create: `data/query_sampling.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append failing query-sampling tests to `tests/test_data.py`**

```python
# ---------------------------------------------------------------------------
# Query sampling
# ---------------------------------------------------------------------------

from data.query_sampling import sample_queries  # noqa: E402

# Task IDs from the spec
TASK_POINT_TRACK = 0
TASK_DEPTH = 1
TASK_POINT_CLOUD = 2
TASK_EXTRINSICS = 3
TASK_INTRINSICS = 4


def _fake_anno(T=16, P=20, H=64, W=64):
    rng = np.random.default_rng(0)
    trajs_2d = (rng.random((T, P, 2)).astype(np.float32)
                * np.array([W - 1, H - 1], dtype=np.float32))
    trajs_3d = rng.standard_normal((T, P, 3)).astype(np.float32)
    trajs_3d[..., 2] = np.abs(trajs_3d[..., 2]) + 0.5
    visibilities = np.ones((T, P), dtype=np.float32)  # always visible -> easy
    depth = rng.random((T, H, W)).astype(np.float32) + 0.5
    normals = rng.standard_normal((T, H, W, 3)).astype(np.float32)
    return trajs_2d, trajs_3d, visibilities, depth, normals


def test_sample_queries_shapes_and_ranges():
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20, H=64, W=64)
    rng = np.random.default_rng(42)
    out = sample_queries(
        num_queries=256, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(0.4, 0.3, 0.15, 0.10, 0.05),
        rng=rng,
    )
    assert out["coords"].shape == (256, 2)
    assert out["t_src"].shape == (256,)
    assert out["t_tgt"].shape == (256,)
    assert out["t_cam"].shape == (256,)
    assert out["task_id"].shape == (256,)
    assert out["coords"].min() >= 0.0 and out["coords"].max() <= 1.0
    assert out["t_src"].min() >= 0 and out["t_src"].max() < 16
    assert out["t_tgt"].min() >= 0 and out["t_tgt"].max() < 16
    assert out["t_cam"].min() >= 0 and out["t_cam"].max() < 16
    assert set(np.unique(out["task_id"]).tolist()) <= {0, 1, 2, 3, 4}


def test_sample_queries_task_mix_distribution():
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(1)
    mix = (0.4, 0.3, 0.15, 0.10, 0.05)
    out = sample_queries(
        num_queries=2000, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=mix, rng=rng,
    )
    counts = np.bincount(out["task_id"], minlength=5)
    fracs = counts / counts.sum()
    # Allow generous tolerance — remainders go to task 0
    np.testing.assert_allclose(fracs[1:], mix[1:], atol=0.02)
    assert fracs[0] >= mix[0] - 0.02  # may be a bit higher from remainders


def test_sample_queries_point_track_invariants():
    """For task 0 queries, t_src must be fixed per trajectory and t_tgt == t_cam."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(2)
    out = sample_queries(
        num_queries=500, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(1.0, 0.0, 0.0, 0.0, 0.0),
        rng=rng,
    )
    assert (out["task_id"] == 0).all()
    np.testing.assert_array_equal(out["t_tgt"], out["t_cam"])


def test_sample_queries_depth_task_invariants():
    """For task 1 queries, t_src == t_tgt == t_cam."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(3)
    out = sample_queries(
        num_queries=300, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(0.0, 1.0, 0.0, 0.0, 0.0),
        rng=rng,
    )
    assert (out["task_id"] == 1).all()
    np.testing.assert_array_equal(out["t_src"], out["t_tgt"])
    np.testing.assert_array_equal(out["t_src"], out["t_cam"])


def test_sample_queries_extrinsics_task_invariants():
    """For task 3 queries, t_src is the anchor (constant) and t_tgt == t_cam."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(4)
    out = sample_queries(
        num_queries=200, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(0.0, 0.0, 0.0, 1.0, 0.0),
        rng=rng,
    )
    assert (out["task_id"] == 3).all()
    # t_src is fixed at the anchor frame
    assert len(np.unique(out["t_src"])) == 1
    np.testing.assert_array_equal(out["t_tgt"], out["t_cam"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` on `data.query_sampling`.

- [ ] **Step 3: Create `data/query_sampling.py`**

```python
"""Mixed-task query sampling for D4RT.

For each training sample, we emit `num_queries` query tuples
(coords, t_src, t_tgt, t_cam) drawn across five task types so the model trains
its unified query interface on the full task palette in every step.

Per-task emission rules follow the spec §5 table.
"""

import numpy as np


# Task IDs (kept as plain ints — they're dataset-side metadata, not consumed by
# the model forward pass).
TASK_POINT_TRACK = 0
TASK_DEPTH = 1
TASK_POINT_CLOUD = 2
TASK_EXTRINSICS = 3
TASK_INTRINSICS = 4


def _split_quota(total: int, mix):
    """Split `total` queries by `mix` ratios; round to ints, remainder -> task 0."""
    assert len(mix) == 5
    raw = np.array(mix, dtype=np.float64) * total
    quotas = np.floor(raw).astype(np.int64)
    remainder = total - int(quotas.sum())
    quotas[0] += remainder
    return quotas.tolist()


def _emit_point_track(quota, num_frames, img_size,
                       trajs_2d, visibilities, rng):
    """Emit point-tracking queries: one trajectory provides T queries (one per
    target frame) with t_src fixed at the trajectory's first visible frame.
    """
    T, P, _ = trajs_2d.shape
    coords = np.zeros((quota, 2), dtype=np.float32)
    t_src = np.zeros((quota,), dtype=np.int64)
    t_tgt = np.zeros((quota,), dtype=np.int64)
    t_cam = np.zeros((quota,), dtype=np.int64)
    traj_idx = -np.ones((quota,), dtype=np.int64)

    # Pick valid trajectories (visible in at least one frame)
    visible_any = visibilities.sum(axis=0) > 0
    valid_traj_idxs = np.where(visible_any)[0]
    if len(valid_traj_idxs) == 0:
        # No tracks — fall back to filling with depth-style queries
        for i in range(quota):
            t = int(rng.integers(0, T))
            u = float(rng.random())
            v = float(rng.random())
            coords[i] = (u, v)
            t_src[i] = t
            t_tgt[i] = t
            t_cam[i] = t
        return coords, t_src, t_tgt, t_cam, traj_idx

    filled = 0
    while filled < quota:
        p = int(rng.choice(valid_traj_idxs))
        # First visible frame in window
        first_vis = int(np.argmax(visibilities[:, p] > 0))
        traj_uv_px = trajs_2d[first_vis, p]   # (2,) in pixel coords
        u = float(traj_uv_px[0] / max(img_size - 1, 1))
        v = float(traj_uv_px[1] / max(img_size - 1, 1))
        u = float(np.clip(u, 0.0, 1.0))
        v = float(np.clip(v, 0.0, 1.0))
        remain = quota - filled
        take = min(T, remain)
        for k in range(take):
            coords[filled + k] = (u, v)
            t_src[filled + k] = first_vis
            t_tgt[filled + k] = k
            t_cam[filled + k] = k
            traj_idx[filled + k] = p
        filled += take

    return coords, t_src, t_tgt, t_cam, traj_idx


def _emit_depth(quota, num_frames, rng):
    """task 1: t_src = t_tgt = t_cam = t. Random pixel, random frame."""
    t = rng.integers(0, num_frames, size=quota).astype(np.int64)
    coords = rng.random((quota, 2)).astype(np.float32)
    return coords, t, t.copy(), t.copy(), -np.ones((quota,), dtype=np.int64)


def _emit_point_cloud(quota, num_frames, anchor, rng):
    """task 2: t_src = t_tgt = t, t_cam = anchor."""
    t = rng.integers(0, num_frames, size=quota).astype(np.int64)
    coords = rng.random((quota, 2)).astype(np.float32)
    t_cam = np.full((quota,), anchor, dtype=np.int64)
    return coords, t, t.copy(), t_cam, -np.ones((quota,), dtype=np.int64)


def _emit_extrinsics(quota, num_frames, anchor, rng):
    """task 3: t_src = anchor, t_tgt = t_cam = random."""
    t_src = np.full((quota,), anchor, dtype=np.int64)
    t_tt = rng.integers(0, num_frames, size=quota).astype(np.int64)
    coords = rng.random((quota, 2)).astype(np.float32)
    return coords, t_src, t_tt, t_tt.copy(), -np.ones((quota,), dtype=np.int64)


def _emit_intrinsics(quota, num_frames, rng):
    """task 4: same tuple as depth (task 1), separate task_id for diagnostics."""
    t = rng.integers(0, num_frames, size=quota).astype(np.int64)
    coords = rng.random((quota, 2)).astype(np.float32)
    return coords, t, t.copy(), t.copy(), -np.ones((quota,), dtype=np.int64)


def sample_queries(
    num_queries: int,
    num_frames: int,
    img_size: int,
    trajs_2d: np.ndarray,
    trajs_3d: np.ndarray,
    visibilities: np.ndarray,
    depth: np.ndarray,
    normals: np.ndarray,
    task_mix=(0.4, 0.3, 0.15, 0.10, 0.05),
    rng: np.random.Generator = None,
) -> dict:
    """Sample a mixed batch of queries spanning all five tasks.

    Returns a dict with:
        coords    (N, 2)     float32 in [0, 1]
        t_src     (N,)       int64
        t_tgt     (N,)       int64
        t_cam     (N,)       int64
        task_id   (N,)       int64
        query_meta dict with extra info targets.py needs (traj_idx per query)
    """
    if rng is None:
        rng = np.random.default_rng()

    quotas = _split_quota(num_queries, task_mix)
    anchor = num_frames // 2

    pieces = []
    traj_idx_all = []
    task_ids = []

    # task 0: point track
    if quotas[0] > 0:
        c, ts, tt, tc, ti = _emit_point_track(
            quotas[0], num_frames, img_size, trajs_2d, visibilities, rng)
        pieces.append((c, ts, tt, tc))
        traj_idx_all.append(ti)
        task_ids.append(np.full((quotas[0],), TASK_POINT_TRACK, dtype=np.int64))

    # task 1: depth
    if quotas[1] > 0:
        c, ts, tt, tc, ti = _emit_depth(quotas[1], num_frames, rng)
        pieces.append((c, ts, tt, tc))
        traj_idx_all.append(ti)
        task_ids.append(np.full((quotas[1],), TASK_DEPTH, dtype=np.int64))

    # task 2: point cloud
    if quotas[2] > 0:
        c, ts, tt, tc, ti = _emit_point_cloud(quotas[2], num_frames, anchor, rng)
        pieces.append((c, ts, tt, tc))
        traj_idx_all.append(ti)
        task_ids.append(np.full((quotas[2],), TASK_POINT_CLOUD, dtype=np.int64))

    # task 3: extrinsics
    if quotas[3] > 0:
        c, ts, tt, tc, ti = _emit_extrinsics(quotas[3], num_frames, anchor, rng)
        pieces.append((c, ts, tt, tc))
        traj_idx_all.append(ti)
        task_ids.append(np.full((quotas[3],), TASK_EXTRINSICS, dtype=np.int64))

    # task 4: intrinsics
    if quotas[4] > 0:
        c, ts, tt, tc, ti = _emit_intrinsics(quotas[4], num_frames, rng)
        pieces.append((c, ts, tt, tc))
        traj_idx_all.append(ti)
        task_ids.append(np.full((quotas[4],), TASK_INTRINSICS, dtype=np.int64))

    coords = np.concatenate([p[0] for p in pieces], axis=0)
    t_src = np.concatenate([p[1] for p in pieces], axis=0)
    t_tgt = np.concatenate([p[2] for p in pieces], axis=0)
    t_cam = np.concatenate([p[3] for p in pieces], axis=0)
    task_id = np.concatenate(task_ids, axis=0)
    traj_idx = np.concatenate(traj_idx_all, axis=0)

    coords = np.clip(coords, 0.0, 1.0)

    return {
        "coords": coords,
        "t_src": t_src,
        "t_tgt": t_tgt,
        "t_cam": t_cam,
        "task_id": task_id,
        "query_meta": {"traj_idx": traj_idx, "anchor": anchor},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 18 tests pass (13 existing + 5 new).

- [ ] **Step 5: Commit**

```bash
git add data/query_sampling.py tests/test_data.py
git commit -m "feat(data): mixed-task query sampling (5 task types)"
```

---

## Task 7: Target generation

**Files:**
- Create: `data/targets.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append failing target-generation tests to `tests/test_data.py`**

```python
# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

from data.targets import build_targets  # noqa: E402


def _make_inputs(T=16, P=20, H=64, W=64, img_size=64, num_queries=128):
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=T, P=P, H=H, W=W)
    K = np.array([[img_size * 0.8, 0, img_size / 2],
                  [0, img_size * 0.8, img_size / 2],
                  [0, 0, 1]], dtype=np.float32)
    intrinsics = np.broadcast_to(K, (T, 3, 3)).copy()
    extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()
    rng = np.random.default_rng(7)
    q = sample_queries(num_queries=num_queries, num_frames=T, img_size=img_size,
                       trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
                       depth=depth, normals=normals,
                       task_mix=(0.4, 0.3, 0.15, 0.10, 0.05), rng=rng)
    return q, trajs_2d, trajs_3d, vis, depth, normals, intrinsics, extrinsics


def test_build_targets_keys_and_shapes():
    q, trajs_2d, trajs_3d, vis, depth, normals, K, E = _make_inputs(num_queries=128)
    tgt = build_targets(
        coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
        task_id=q["task_id"], query_meta=q["query_meta"],
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        intrinsics=K, extrinsics=E, img_size=64,
    )
    required = {"pos_3d", "pos_2d", "visibility", "displacement", "normal",
                "mask_3d", "mask_2d", "mask_vis", "mask_disp", "mask_normal"}
    assert required <= set(tgt.keys())
    N = 128
    assert tgt["pos_3d"].shape == (N, 3)
    assert tgt["pos_2d"].shape == (N, 2)
    assert tgt["visibility"].shape == (N,)
    assert tgt["displacement"].shape == (N, 3)
    assert tgt["normal"].shape == (N, 3)
    for m in ("mask_3d", "mask_2d", "mask_vis", "mask_disp", "mask_normal"):
        assert tgt[m].shape == (N,)
        assert tgt[m].dtype == torch.float32
        assert tgt[m].min() >= 0.0 and tgt[m].max() <= 1.0


def test_build_targets_displacement_only_for_track_queries():
    q, trajs_2d, trajs_3d, vis, depth, normals, K, E = _make_inputs(num_queries=512)
    tgt = build_targets(
        coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
        task_id=q["task_id"], query_meta=q["query_meta"],
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        intrinsics=K, extrinsics=E, img_size=64,
    )
    # mask_disp == 1 only where task_id == 0 (and pos_3d valid both ends)
    is_track = torch.from_numpy(q["task_id"] == 0)
    # Wherever mask_disp == 1, it must be a track query
    assert (tgt["mask_disp"][~is_track] == 0).all()


def test_build_targets_pos_2d_normalized():
    q, trajs_2d, trajs_3d, vis, depth, normals, K, E = _make_inputs(num_queries=128)
    tgt = build_targets(
        coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
        task_id=q["task_id"], query_meta=q["query_meta"],
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        intrinsics=K, extrinsics=E, img_size=64,
    )
    # pos_2d should mostly be in [-eps, 1+eps] for valid queries
    valid = tgt["mask_3d"] > 0
    pos_2d_valid = tgt["pos_2d"][valid]
    # Allow some out-of-bounds (off-screen projections); just check ballpark
    in_bounds_frac = ((pos_2d_valid >= -0.5) & (pos_2d_valid <= 1.5)).all(dim=-1).float().mean()
    assert in_bounds_frac > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` on `data.targets`.

- [ ] **Step 3: Create `data/targets.py`**

```python
"""Build the `targets` dict consumed by D4RTLoss from PointOdyssey annotations.

Per-task derivation rules follow spec §6. All output tensors are torch.float32
(masks too) so they can be stacked by `collate_fn` without dtype conversion.
"""

from typing import Optional

import numpy as np
import torch


def _backproject(u_pix: np.ndarray, v_pix: np.ndarray, z: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    """Back-project pixel coords + depth into camera-frame 3D points.

    Uses the OpenCV convention: +Z forward, +X right, +Y down.

    Args:
        u_pix, v_pix: (N,) pixel coords.
        z: (N,) depth values at those pixels.
        K: (N, 3, 3) per-query intrinsics (broadcast outside).

    Returns:
        (N, 3) camera-frame 3D points.
    """
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    x = (u_pix - cx) * z / fx
    y = (v_pix - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _project(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame 3D points into pixel coords. (N, 3) -> (N, 2)."""
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    z = np.clip(pts[:, 2], 1e-6, None)
    u = pts[:, 0] / z * fx + cx
    v = pts[:, 1] / z * fy + cy
    return np.stack([u, v], axis=-1)


def _per_query_K(intrinsics: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Resolve per-query (N, 3, 3) intrinsics from either static or per-frame K."""
    if intrinsics.ndim == 2:
        return np.broadcast_to(intrinsics, (t.shape[0], 3, 3)).copy()
    return intrinsics[t]


def _per_query_E(extrinsics: Optional[np.ndarray], t: np.ndarray, default_eye=True):
    if extrinsics is None:
        if default_eye:
            return np.broadcast_to(np.eye(4, dtype=np.float32),
                                    (t.shape[0], 4, 4)).copy()
        return None
    return extrinsics[t]


def build_targets(
    coords: np.ndarray,           # (N, 2) normalized [0, 1]
    t_src: np.ndarray,            # (N,)
    t_tgt: np.ndarray,
    t_cam: np.ndarray,
    task_id: np.ndarray,
    query_meta: dict,
    trajs_2d: np.ndarray,         # (T, P, 2) pixel coords
    trajs_3d: np.ndarray,         # (T, P, 3) world coords
    visibilities: np.ndarray,     # (T, P)
    depth: np.ndarray,            # (T, H, W)
    normals: np.ndarray,          # (T, H, W, 3)
    intrinsics: np.ndarray,       # (3, 3) or (T, 3, 3)
    extrinsics: Optional[np.ndarray],   # (T, 4, 4) or None
    img_size: int,
) -> dict:
    """Build the loss-ready targets dict.

    Returns torch tensors. See spec §6 for per-task derivation rules.
    """
    N = coords.shape[0]
    T, H, W = depth.shape
    traj_idx = query_meta["traj_idx"]    # (N,) -1 for non-track

    # Pixel coords (clamped, integer for index lookup).
    u_pix_f = coords[:, 0] * (img_size - 1)
    v_pix_f = coords[:, 1] * (img_size - 1)
    u_idx = np.clip(np.round(u_pix_f).astype(np.int64), 0, W - 1)
    v_idx = np.clip(np.round(v_pix_f).astype(np.int64), 0, H - 1)

    pos_3d = np.zeros((N, 3), dtype=np.float32)
    pos_2d = np.zeros((N, 2), dtype=np.float32)
    visibility = np.zeros((N,), dtype=np.float32)
    displacement = np.zeros((N, 3), dtype=np.float32)
    normal = np.zeros((N, 3), dtype=np.float32)
    mask_3d = np.zeros((N,), dtype=np.float32)
    mask_2d = np.zeros((N,), dtype=np.float32)
    mask_vis = np.zeros((N,), dtype=np.float32)
    mask_disp = np.zeros((N,), dtype=np.float32)
    mask_normal = np.zeros((N,), dtype=np.float32)

    # ------------------------------------------------------------
    # Task 0: point track
    # ------------------------------------------------------------
    is_track = task_id == 0
    if is_track.any():
        idxs = np.where(is_track)[0]
        p = traj_idx[idxs]                  # (n,)
        ttgt = t_tgt[idxs]
        tcam = t_cam[idxs]
        tsrc = t_src[idxs]

        # World-frame positions at t_tgt, then transform with extrinsics[t_cam].
        world_tgt = trajs_3d[ttgt, p]       # (n, 3)
        world_src = trajs_3d[tsrc, p]
        E_cam = _per_query_E(extrinsics, tcam)   # (n, 4, 4)

        # world -> camera-t_cam: (E @ [X;1])[:3]
        ones = np.ones((world_tgt.shape[0], 1), dtype=np.float32)
        w_tgt_h = np.concatenate([world_tgt, ones], axis=1)
        w_src_h = np.concatenate([world_src, ones], axis=1)
        cam_tgt = np.einsum("nij,nj->ni", E_cam, w_tgt_h)[:, :3]
        cam_src = np.einsum("nij,nj->ni", E_cam, w_src_h)[:, :3]

        pos_3d[idxs] = cam_tgt
        displacement[idxs] = cam_tgt - cam_src

        # 2D projection at t_tgt
        K_tgt = _per_query_K(intrinsics, ttgt)
        pix = _project(cam_tgt, K_tgt) / (img_size - 1)
        pos_2d[idxs] = pix.astype(np.float32)

        # Visibility from annotations
        vis_vals = visibilities[ttgt, p].astype(np.float32)
        visibility[idxs] = vis_vals
        mask_vis[idxs] = 1.0

        # Mask 3D where Z is positive
        valid_3d = cam_tgt[:, 2] > 0
        valid_src = cam_src[:, 2] > 0
        mask_3d[idxs] = valid_3d.astype(np.float32)
        mask_2d[idxs] = (valid_3d & (vis_vals > 0.5)).astype(np.float32)
        mask_disp[idxs] = (valid_3d & valid_src).astype(np.float32)

        # Normal at t_src pixel
        traj_uv_src = trajs_2d[tsrc, p]   # (n, 2) pixel
        u_n = np.clip(np.round(traj_uv_src[:, 0]).astype(np.int64), 0, W - 1)
        v_n = np.clip(np.round(traj_uv_src[:, 1]).astype(np.int64), 0, H - 1)
        normal[idxs] = normals[tsrc, v_n, u_n]
        mask_normal[idxs] = 1.0

    # ------------------------------------------------------------
    # Tasks 1 & 4: depth / intrinsics (degenerate t_src=t_tgt=t_cam=t)
    # ------------------------------------------------------------
    is_d14 = (task_id == 1) | (task_id == 4)
    if is_d14.any():
        idxs = np.where(is_d14)[0]
        t = t_tgt[idxs]
        u_i = u_idx[idxs]
        v_i = v_idx[idxs]
        z = depth[t, v_i, u_i]
        K = _per_query_K(intrinsics, t)
        pts = _backproject(u_pix_f[idxs], v_pix_f[idxs], z, K)
        pos_3d[idxs] = pts.astype(np.float32)

        pix = _project(pts, K) / (img_size - 1)
        pos_2d[idxs] = pix.astype(np.float32)

        valid = (z > 0) & np.isfinite(z)
        visibility[idxs] = valid.astype(np.float32)
        mask_3d[idxs] = valid.astype(np.float32)
        mask_2d[idxs] = valid.astype(np.float32)
        mask_vis[idxs] = 1.0
        # displacement = 0, mask_disp = 0 (default zeros — correct)
        normal[idxs] = normals[t, v_i, u_i]
        mask_normal[idxs] = 1.0

    # ------------------------------------------------------------
    # Task 2: point cloud  (t_src = t_tgt = t,  t_cam = anchor)
    # Task 3: extrinsics   (t_src = anchor,    t_tgt = t_cam = t)
    # Both back-project at t_src, then transform from cam_t_src -> cam_t_cam
    # via E_cam @ inv(E_src).
    # ------------------------------------------------------------
    is_23 = (task_id == 2) | (task_id == 3)
    if is_23.any():
        idxs = np.where(is_23)[0]
        tsrc = t_src[idxs]
        tcam = t_cam[idxs]
        u_i = u_idx[idxs]
        v_i = v_idx[idxs]
        z = depth[tsrc, v_i, u_i]
        K_src = _per_query_K(intrinsics, tsrc)
        cam_src_pts = _backproject(u_pix_f[idxs], v_pix_f[idxs], z, K_src)  # (n,3)

        # Transform to cam_t_cam.
        E_src = _per_query_E(extrinsics, tsrc)
        E_cam = _per_query_E(extrinsics, tcam)
        # E_src maps world->cam_src; we want cam_src->cam_cam = E_cam @ inv(E_src)
        E_src_inv = np.linalg.inv(E_src)
        T_rel = np.einsum("nij,njk->nik", E_cam, E_src_inv)

        ones = np.ones((cam_src_pts.shape[0], 1), dtype=np.float32)
        h = np.concatenate([cam_src_pts, ones], axis=1)
        cam_cam_pts = np.einsum("nij,nj->ni", T_rel, h)[:, :3]
        pos_3d[idxs] = cam_cam_pts.astype(np.float32)

        # Project into camera at t_tgt for pos_2d
        ttgt = t_tgt[idxs]
        K_tgt = _per_query_K(intrinsics, ttgt)
        pix = _project(cam_cam_pts, K_tgt) / (img_size - 1)
        pos_2d[idxs] = pix.astype(np.float32)

        valid = (z > 0) & np.isfinite(z) & (cam_cam_pts[:, 2] > 0)
        visibility[idxs] = valid.astype(np.float32)
        mask_3d[idxs] = valid.astype(np.float32)
        mask_2d[idxs] = valid.astype(np.float32)
        mask_vis[idxs] = 1.0
        normal[idxs] = normals[tsrc, v_i, u_i]
        mask_normal[idxs] = 1.0

    return {
        "pos_3d": torch.from_numpy(pos_3d),
        "pos_2d": torch.from_numpy(pos_2d),
        "visibility": torch.from_numpy(visibility),
        "displacement": torch.from_numpy(displacement),
        "normal": torch.from_numpy(normal),
        "mask_3d": torch.from_numpy(mask_3d),
        "mask_2d": torch.from_numpy(mask_2d),
        "mask_vis": torch.from_numpy(mask_vis),
        "mask_disp": torch.from_numpy(mask_disp),
        "mask_normal": torch.from_numpy(mask_normal),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 21 tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/targets.py tests/test_data.py
git commit -m "feat(data): per-task target generation for D4RTLoss"
```

---

## Task 8: PointOdyssey dataset

**Files:**
- Create: `data/pointodyssey.py`
- Modify: `data/__init__.py`
- Modify: `tests/test_data.py` (append)

- [ ] **Step 1: Append the PointOdyssey smoke test to `tests/test_data.py`**

```python
# ---------------------------------------------------------------------------
# PointOdysseyDataset (integration)
# ---------------------------------------------------------------------------

from data import PointOdysseyDataset  # noqa: E402
from data.augmentations import AugmentationConfig as _AC, VideoAugmentation as _VA  # noqa: E402


def test_pointodyssey_smoke(tmp_path):
    make_fake_pointodyssey(tmp_path, num_sequences=2, num_frames=16,
                            img_h=64, img_w=64, num_trajs=30)
    ds = PointOdysseyDataset(
        data_root=str(tmp_path),
        split="train",
        num_frames=8,
        img_size=32,
        num_queries=64,
        transform=_VA(_AC(blur_prob=0.0)),
    )
    assert len(ds) == 2
    sample = ds[0]

    assert sample["video"].shape == (8, 32, 32, 3)
    assert sample["video"].dtype == torch.float32
    assert sample["video"].min() >= 0.0 and sample["video"].max() <= 1.0

    assert sample["coords"].shape == (64, 2)
    assert sample["t_src"].shape == (64,)
    assert sample["t_tgt"].shape == (64,)
    assert sample["t_cam"].shape == (64,)
    assert sample["aspect_ratio"].shape == (2,)

    for key in ("pos_3d", "pos_2d", "visibility", "displacement", "normal",
                "mask_3d", "mask_2d", "mask_vis", "mask_disp", "mask_normal"):
        assert key in sample["targets"], f"missing target key: {key}"
    assert sample["targets"]["pos_3d"].shape == (64, 3)
    assert sample["targets"]["mask_3d"].shape == (64,)


def test_pointodyssey_drops_short_sequences(tmp_path):
    # 1 long seq + 1 short seq; ask for num_frames longer than the short one
    make_fake_pointodyssey(tmp_path, num_sequences=1, num_frames=32)
    # Add a short sequence manually
    short_dir = tmp_path / "train" / "seq_short"
    short_dir.mkdir(parents=True)
    (short_dir / "rgbs").mkdir()
    # Only 4 frames — too short for num_frames=16
    for t in range(4):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        _PILImage.fromarray(arr).save(short_dir / "rgbs" / f"rgb_{t:05d}.jpg")
    # No depths/normals etc — the scanner shouldn't even attempt to read

    ds = PointOdysseyDataset(
        data_root=str(tmp_path), split="train",
        num_frames=16, img_size=32, num_queries=32, transform=None,
    )
    # Only the long sequence survives the length filter
    assert len(ds) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` on `PointOdysseyDataset`.

- [ ] **Step 3: Create `data/pointodyssey.py`**

```python
"""PointOdysseyDataset: the fully-implemented dataset for D4RT training.

See docs/superpowers/specs/2026-05-25-data-module-design.md (§4) for the
expected on-disk layout and __getitem__ contract.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.augmentations import AugmentationConfig, TemporalSubsampling
from data.base import compute_aspect_ratio, resize_frames_square, to_float32_normalized
from data.query_sampling import sample_queries
from data.targets import build_targets


REQUIRED_ANNO_KEYS = {"trajs_2d", "trajs_3d", "visibilities"}


class PointOdysseyDataset(Dataset):
    """PointOdyssey dataset producing D4RT-shaped training samples.

    Args:
        data_root: Root directory containing `{split}/` subdirectories.
        split: 'train' | 'val' | 'test' | 'sample'.
        num_frames: Frames per clip after subsampling.
        img_size: Square resize target.
        num_queries: Queries per sample.
        transform: Optional VideoAugmentation instance (applied after resize).
        task_mix: Per-task query mix (point_track, depth, point_cloud, extrinsics, intrinsics).
        min_visible_frac: Min visibility fraction for a trajectory to be used as a track query.
        rng_seed: Optional base seed; per-sample RNG combines this with idx.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 48,
        img_size: int = 256,
        num_queries: int = 2048,
        transform=None,
        task_mix=(0.4, 0.3, 0.15, 0.10, 0.05),
        min_visible_frac: float = 0.5,
        rng_seed: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_queries = num_queries
        self.transform = transform
        self.task_mix = task_mix
        self.min_visible_frac = min_visible_frac
        self.rng_seed = rng_seed
        self._temporal = TemporalSubsampling(AugmentationConfig())

        split_dir = self.data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"PointOdyssey split directory not found: {split_dir}"
            )

        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            rgbs = seq_dir / "rgbs"
            if not rgbs.is_dir():
                continue
            frame_count = sum(1 for f in rgbs.iterdir() if f.suffix.lower() in {".jpg", ".png"})
            if frame_count >= num_frames:
                sequences.append((seq_dir, frame_count))

        if not sequences:
            raise RuntimeError(
                f"No sequences in {split_dir} with >= {num_frames} frames."
            )

        # Probe one sequence to validate anno schema.
        probe_anno = sequences[0][0] / "anno.npz"
        if probe_anno.exists():
            with np.load(probe_anno) as a:
                missing = REQUIRED_ANNO_KEYS - set(a.files)
                if missing:
                    raise RuntimeError(
                        f"{probe_anno} missing required keys: {missing}"
                    )

        # Probe extrinsics: warn once if missing.
        if not (sequences[0][0] / "extrinsics.npy").exists():
            warnings.warn(
                "No extrinsics.npy in first sequence — falling back to identity "
                "extrinsics. Point-cloud / extrinsics task queries that span "
                "different frames will be masked out.",
                stacklevel=2,
            )

        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_dir, frame_count = self.sequences[idx]

        # Per-sample RNG (deterministic if rng_seed set)
        base_seed = self.rng_seed if self.rng_seed is not None else 0
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng((base_seed * 1_000_003 + worker_id * 9973 + idx) & 0xFFFFFFFF)

        # Pick stride and start frame
        stride = self._temporal.sample_stride(rng)
        span = self.num_frames * stride
        # Bound stride if window doesn't fit
        while span > frame_count and stride > 1:
            stride -= 1
            span = self.num_frames * stride
        max_start = max(0, frame_count - span)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        frame_indices = np.arange(start, start + span, stride)[: self.num_frames]

        # Load RGBs (raw resolution)
        rgbs_dir = seq_dir / "rgbs"
        rgb_files = sorted(rgbs_dir.iterdir())
        raw = []
        for t in frame_indices:
            img = Image.open(rgb_files[int(t)]).convert("RGB")
            raw.append(np.asarray(img))
        raw = np.stack(raw, axis=0)             # (T, H_orig, W_orig, 3) uint8
        orig_h, orig_w = raw.shape[1], raw.shape[2]
        aspect_ratio = compute_aspect_ratio(orig_h, orig_w)

        # Resize to square
        frames = resize_frames_square(raw, self.img_size)   # (T, S, S, 3) uint8
        video = to_float32_normalized(frames)               # (T, S, S, 3) float32 in [0, 1]

        # Photometric augmentation (NO geometric — targets pass through unchanged)
        if self.transform is not None:
            video = self.transform(video)

        # Load depth / normals at original resolution, then resize to img_size
        depth_files = sorted((seq_dir / "depths").iterdir())
        normal_files = sorted((seq_dir / "normals").iterdir()) if (seq_dir / "normals").is_dir() else None

        depth_stack = []
        normal_stack = []
        for t in frame_indices:
            d = np.load(depth_files[int(t)])
            # Resize depth to img_size via nearest-neighbor (cheap, avoids smoothing artifacts at edges)
            d_img = Image.fromarray(d.astype(np.float32), mode="F").resize(
                (self.img_size, self.img_size), Image.NEAREST)
            depth_stack.append(np.asarray(d_img, dtype=np.float32))
            if normal_files is not None:
                n = np.load(normal_files[int(t)])
                # Resize per-channel
                n_img = np.stack([
                    np.asarray(Image.fromarray(n[..., c], mode="F").resize(
                        (self.img_size, self.img_size), Image.NEAREST),
                        dtype=np.float32)
                    for c in range(3)
                ], axis=-1)
                normal_stack.append(n_img)
            else:
                normal_stack.append(np.zeros((self.img_size, self.img_size, 3),
                                              dtype=np.float32))
        depth = np.stack(depth_stack, axis=0)        # (T, S, S)
        normals = np.stack(normal_stack, axis=0)     # (T, S, S, 3)

        # Annotations
        anno_path = seq_dir / "anno.npz"
        with np.load(anno_path) as a:
            trajs_2d_full = a["trajs_2d"].astype(np.float32)       # (T_full, P, 2)
            trajs_3d_full = a["trajs_3d"].astype(np.float32)
            visibilities_full = a["visibilities"].astype(np.float32)

        # Slice annotations to the window AND rescale 2D pixel coords to img_size
        sx = self.img_size / orig_w
        sy = self.img_size / orig_h
        trajs_2d = trajs_2d_full[frame_indices].copy()
        trajs_2d[..., 0] *= sx
        trajs_2d[..., 1] *= sy
        trajs_3d = trajs_3d_full[frame_indices].copy()
        visibilities = visibilities_full[frame_indices].copy()

        # Camera matrices
        K_path = seq_dir / "intrinsics.npy"
        K_full = np.load(K_path).astype(np.float32)
        if K_full.ndim == 2:
            K_window = K_full.copy()
            K_window = np.broadcast_to(K_window, (len(frame_indices), 3, 3)).copy()
        else:
            K_window = K_full[frame_indices].copy()
        # Rescale intrinsics for the resize
        K_window[:, 0, 0] *= sx
        K_window[:, 0, 2] *= sx
        K_window[:, 1, 1] *= sy
        K_window[:, 1, 2] *= sy

        E_path = seq_dir / "extrinsics.npy"
        if E_path.exists():
            E_full = np.load(E_path).astype(np.float32)
            E_window = E_full[frame_indices].copy()
        else:
            E_window = None

        # Sample queries
        q = sample_queries(
            num_queries=self.num_queries,
            num_frames=self.num_frames,
            img_size=self.img_size,
            trajs_2d=trajs_2d,
            trajs_3d=trajs_3d,
            visibilities=visibilities,
            depth=depth,
            normals=normals,
            task_mix=self.task_mix,
            rng=rng,
        )

        # Build targets
        targets = build_targets(
            coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
            task_id=q["task_id"], query_meta=q["query_meta"],
            trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=visibilities,
            depth=depth, normals=normals,
            intrinsics=K_window, extrinsics=E_window, img_size=self.img_size,
        )

        return {
            "video": torch.from_numpy(video),                    # (T, S, S, 3) float32
            "coords": torch.from_numpy(q["coords"]).float(),     # (N, 2)
            "t_src": torch.from_numpy(q["t_src"]).long(),
            "t_tgt": torch.from_numpy(q["t_tgt"]).long(),
            "t_cam": torch.from_numpy(q["t_cam"]).long(),
            "aspect_ratio": torch.from_numpy(aspect_ratio).float(),
            "targets": targets,
        }
```

- [ ] **Step 4: Update `data/__init__.py` to export PointOdysseyDataset**

Overwrite `data/__init__.py`:

```python
"""D4RT data loading package."""

from data.collate import collate_fn
from data.pointodyssey import PointOdysseyDataset
from data.stubs import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset

__all__ = [
    "PointOdysseyDataset",
    "VideoDataset",
    "KubricDataset",
    "SintelDataset",
    "ScanNetDataset",
    "collate_fn",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 23 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/pointodyssey.py data/__init__.py tests/test_data.py
git commit -m "feat(data): PointOdysseyDataset (fully implemented)"
```

---

## Task 9: Loss-compat integration test

**Files:**
- Modify: `tests/test_data.py` (append)

This task wires the dataset's batch dict through the actual `D4RTLoss` to prove the targets dict satisfies the loss contract. This is the real safety net.

- [ ] **Step 1: Append the loss-compat test**

```python
# ---------------------------------------------------------------------------
# Integration: collate + D4RTLoss
# ---------------------------------------------------------------------------

from losses import D4RTLoss  # noqa: E402


def test_collate_and_loss_compat(tmp_path):
    make_fake_pointodyssey(tmp_path, num_sequences=2, num_frames=16)
    ds = PointOdysseyDataset(
        data_root=str(tmp_path), split="train",
        num_frames=8, img_size=32, num_queries=64, transform=None,
    )
    batch = collate_fn([ds[0], ds[1]])

    B, N = 2, 64
    predictions = {
        "pos_3d": torch.randn(B, N, 3, requires_grad=True),
        "pos_2d": torch.randn(B, N, 2),
        "visibility": torch.randn(B, N, 1),
        "displacement": torch.randn(B, N, 3),
        "normal": torch.randn(B, N, 3),
        "confidence": torch.sigmoid(torch.randn(B, N, 1)),
    }

    losses = D4RTLoss()(predictions, batch["targets"])
    assert "loss" in losses
    assert torch.isfinite(losses["loss"]).item()
    losses["loss"].backward()
    assert predictions["pos_3d"].grad is not None
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 24 tests pass. The whole test file should run in <5 seconds.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data.py
git commit -m "test(data): collate -> D4RTLoss integration smoke"
```

---

## Task 10: Wire PointOdyssey into train.py

**Files:**
- Modify: `train.py`

- [ ] **Step 1: Read current train.py state around dataset selection**

```bash
grep -n "kubric\|scannet\|sintel\|dataset" train.py | head -30
```

Confirm lines:
- Imports: `from data import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset, collate_fn` (around line 42)
- argparse choice: `choices=['video', 'kubric', 'sintel', 'scannet']` (around line 110)
- `create_dataloader` branches: `if args.dataset == 'kubric':` ... (around line 187)

- [ ] **Step 2: Patch the import**

Change line 42 of `train.py` from:

```python
from data import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset, collate_fn
```

To:

```python
from data import (
    PointOdysseyDataset,
    VideoDataset,
    KubricDataset,
    SintelDataset,
    ScanNetDataset,
    collate_fn,
)
```

- [ ] **Step 3: Patch the argparse choices**

Find the `--dataset` argparse argument (around line 110) and update its choices.

Current:
```python
    parser.add_argument('--dataset', type=str, default='video',
                        choices=['video', 'kubric', 'sintel', 'scannet'],
                        help='Dataset type')
```

Change to:
```python
    parser.add_argument('--dataset', type=str, default='video',
                        choices=['video', 'kubric', 'sintel', 'scannet', 'pointodyssey'],
                        help='Dataset type')
```

- [ ] **Step 4: Add the PointOdyssey branch in `create_dataloader`**

Insert this branch in `create_dataloader` BEFORE the `if args.dataset == 'kubric':` branch:

```python
    if args.dataset == 'pointodyssey':
        dataset = PointOdysseyDataset(
            args.data_root,
            split='train',
            num_frames=args.num_frames,
            img_size=args.img_size,
            num_queries=args.num_queries,
            transform=transform
        )
    elif args.dataset == 'kubric':
```

(Convert the existing `if` to `elif` so the chain flows naturally.)

- [ ] **Step 5: Smoke-test the wiring**

Build a tiny synthetic PointOdyssey dataset on a temp path and confirm `train.py --dataset pointodyssey ... --steps 0` constructs the dataloader without errors. From a Python shell:

```bash
python -c "
import tempfile, sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_data import make_fake_pointodyssey

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    make_fake_pointodyssey(root, num_sequences=2, num_frames=16)
    # Build the dataloader the way train.py does
    import argparse
    args = argparse.Namespace(
        dataset='pointodyssey', data_root=str(root),
        num_frames=8, img_size=32, num_queries=64, num_workers=0,
        batch_size=1,
    )
    from train import create_dataloader
    dl, _ = create_dataloader(args, rank=0, world_size=1)
    batch = next(iter(dl))
    print('OK', list(batch.keys()))
    assert batch['video'].shape == (1, 8, 32, 32, 3)
    assert 'pos_3d' in batch['targets']
"
```

Expected output (single line):
```
OK ['video', 'coords', 't_src', 't_tgt', 't_cam', 'aspect_ratio', 'targets']
```

- [ ] **Step 6: Commit**

```bash
git add train.py
git commit -m "feat(train): add --dataset pointodyssey choice"
```

---

## Self-Review Notes

**Spec coverage check** (each spec section → task):
- §2 Contract → Task 8 (pointodyssey.py) + Task 9 (loss-compat test).
- §3 Module layout → Tasks 1-8 (one task per file).
- §4 PointOdysseyDataset → Task 8.
- §5 Query sampling → Task 6.
- §6 Target generation → Task 7.
- §7 Augmentation → Task 1.
- §8 Stubs → Task 2.
- §9 Collate → Task 3.
- §10 Testing → Tests live alongside each task; integration tests in Task 9.
- §11 Implementation order → Matches Tasks 1-10 with `train.py` patch at the end.
- §12 Out of scope → Honored (no boundary sampling, no geometric augmentation, stubs not implemented).

**Type/name consistency check:**
- `sample_queries` returns `{"coords", "t_src", "t_tgt", "t_cam", "task_id", "query_meta"}` everywhere it's mentioned. ✓
- `build_targets` takes `coords, t_src, t_tgt, t_cam, task_id, query_meta, trajs_2d, trajs_3d, visibilities, depth, normals, intrinsics, extrinsics, img_size` in Task 7 and is called with the same signature in Task 8. ✓
- `collate_fn` signature: `(batch) -> dict`. Used in Task 3, Task 9, and indirectly via `DataLoader(collate_fn=collate_fn)`. ✓
- `PointOdysseyDataset.__init__` kwargs match `train.py:create_dataloader`'s call site. ✓
- `compute_aspect_ratio(h, w) -> np.ndarray` shape `(2,)`. Used in Task 8. ✓
- Target keys returned by `build_targets` match the keys checked by `test_collate_and_loss_compat` and consumed by `D4RTLoss`. ✓

**No placeholders.** Every code block is complete and runnable.

---

## Execution

Once tasks are complete, the verification is:
```bash
pytest tests/test_data.py -v
```
All 24 tests should pass in under 10 seconds, and `train.py --dataset pointodyssey` should construct a dataloader cleanly against a synthetic fixture.

# D4RT Data Module — Design Spec

**Date:** 2026-05-25
**Status:** Approved (pending user review of written spec)
**Scope:** Implement the `data/` package so `train.py` runs end-to-end against PointOdyssey. Other datasets (Kubric, Sintel, ScanNet, generic Video) ship as stubs that raise `NotImplementedError`.

---

## 1. Goals & Non-Goals

### Goals
- Make `train.py --dataset pointodyssey --data-root /path/to/pointodyssey` work end-to-end with the existing model and loss code.
- Produce `targets` dicts that exercise every component of `D4RTLoss`: 3D, 2D, visibility, displacement, normal, confidence.
- Cover all five task types from `CLAUDE.md` (point-track, depth, point-cloud, extrinsics, intrinsics) within a single training sample (mixed query sampling).
- Provide stubs for the other dataset names `train.py` imports so they fail with clear messages instead of import errors.
- Be testable without downloading the ~170GB PointOdyssey dataset — a synthetic fixture exercises the pipeline.

### Non-Goals (explicitly deferred)
- Boundary-aware query sampling (`30% near depth/motion boundaries`). First pass uses uniform spatial sampling. Add as a follow-up once training stability is verified.
- Geometric augmentation (horizontal flip, random crop, rotation). First pass is photometric + temporal subsampling only — geometric transforms also require remapping `pos_2d`/`pos_3d` targets, which is bug-prone and not needed to start training.
- Full implementations of `KubricDataset`, `SintelDataset`, `ScanNetDataset`, `VideoDataset`. Each ships as a stub that raises with a pointer to PointOdyssey.
- Evaluation-specific (`evaluate.py`) integration. The dataset will support `split='val'`/`'test'` so it can be reused, but evaluate.py is a separate concern.

---

## 2. Contract with `train.py` and `D4RTLoss`

### Imports `train.py` makes (must continue to work)
```python
from data import VideoDataset, KubricDataset, SintelDataset, ScanNetDataset, collate_fn
from data.augmentations import VideoAugmentation, TemporalSubsampling, AugmentationConfig
```
We add `PointOdysseyDataset` to the `data/__init__.py` re-exports and add `'pointodyssey'` to `train.py`'s `--dataset` choices + the `create_dataloader` branch.

### Dataset constructor signature (matches `train.py:create_dataloader`)
```python
DatasetClass(
    data_root,           # positional
    split='train',
    num_frames=...,
    img_size=...,
    num_queries=...,
    transform=...,
)
```

### `__getitem__` output (after `collate_fn`, what `forward_backward_step` consumes)
| Key | Shape | Notes |
|---|---|---|
| `video` | `(B, T, H, W, 3)` float32 | `[0, 1]`; permuted to `(B, C, T, H, W)` before encoder. |
| `coords` | `(B, N, 2)` float32 | Normalized `[0, 1]` along `(u=x/W, v=y/H)`. |
| `t_src` | `(B, N)` int64 | Source frame index for each query. |
| `t_tgt` | `(B, N)` int64 | Target frame index. |
| `t_cam` | `(B, N)` int64 | Camera reference frame index. |
| `aspect_ratio` | `(B, 2)` float32 | `(orig_H/max, orig_W/max)`. |
| `targets` | `dict[str, Tensor]` | See §5. |

### `targets` dict (consumed by `D4RTLoss`)
| Key | Shape | Required? |
|---|---|---|
| `pos_3d` | `(B, N, 3)` float32 | Yes — primary loss. In `t_cam` camera frame. |
| `pos_2d` | `(B, N, 2)` float32 | Yes — normalized `[0, 1]` to match `coords`. |
| `visibility` | `(B, N)` float32 | Yes. |
| `displacement` | `(B, N, 3)` float32 | Yes — non-track queries get zeros + `mask_disp=0`. |
| `normal` | `(B, N, 3)` float32 | Yes. |
| `mask_3d` | `(B, N)` float32 | Yes — gates the primary 3D loss. |
| `mask_2d` | `(B, N)` float32 | Optional (loss falls back to `mask_3d`). |
| `mask_vis` | `(B, N)` float32 | Optional (loss falls back to `mask_3d`). |
| `mask_disp` | `(B, N)` float32 | Required for `loss_disp` to fire. |
| `mask_normal` | `(B, N)` float32 | Required for `loss_normal` to fire. |

---

## 3. Module layout

```
data/
├── __init__.py          # Re-exports
├── base.py              # BaseVideoDataset — resize/normalize/aspect-ratio helpers
├── pointodyssey.py      # PointOdysseyDataset (fully implemented)
├── stubs.py             # VideoDataset, KubricDataset, SintelDataset, ScanNetDataset
├── query_sampling.py    # sample_queries() — mixed-task per-sample sampler
├── targets.py           # build_targets() — turns raw GT into the loss-ready dict
├── augmentations.py     # VideoAugmentation, TemporalSubsampling, AugmentationConfig
└── collate.py           # collate_fn
```

Each file has one responsibility small enough to read top-to-bottom in a single editor screen. Stubs and base are <100 lines each; `pointodyssey.py`, `query_sampling.py`, `targets.py` are the meaty files.

### `train.py` patch
Add `'pointodyssey'` to the `--dataset` argparse choices and add a new branch in `create_dataloader`:
```python
elif args.dataset == 'pointodyssey':
    dataset = PointOdysseyDataset(
        args.data_root, split='train',
        num_frames=args.num_frames, img_size=args.img_size,
        num_queries=args.num_queries, transform=transform,
    )
```

---

## 4. `PointOdysseyDataset` behavior

### Constructor
```python
PointOdysseyDataset(
    data_root,
    split='train',                       # 'train' | 'val' | 'test' | 'sample'
    num_frames=48,
    img_size=256,
    num_queries=2048,
    transform=None,
    task_mix=(0.4, 0.3, 0.15, 0.10, 0.05),  # point_track, depth, point_cloud, extr, intr
    min_visible_frac=0.5,                # reject trajs visible in <50% of window
    rng_seed=None,
)
```

`task_id` is kept separate for depth (id=1) and intrinsics (id=4) even though they emit identical `(u,v,t_src,t_tgt,t_cam)` tuples — purely for dataset-side metadata/diagnostics; it does not enter the model forward pass.

### On-disk assumptions
```
{data_root}/{split}/{sequence_id}/
    rgbs/         rgb_00000.jpg, rgb_00001.jpg, ...
    depths/       depth_00000.npy, ...
    normals/      normal_00000.npy, ...
    anno.npz      keys: trajs_2d (T,P,2), trajs_3d (T,P,3), visibilities (T,P)
    intrinsics.npy   (T, 3, 3) or (3, 3)
    extrinsics.npy   (T, 4, 4) world→camera     # may be absent
```
If `extrinsics.npy` is absent: point-track falls back to using `trajs_3d` directly (treating world == camera-0); point-cloud and extrinsics queries set `mask_3d=0` when `t_src != t_cam`. Logged once on dataset init.

### `__init__`
1. Scan `{data_root}/{split}/` for sequence directories; cache `(sequence_id, frame_count)` list.
2. Drop sequences shorter than `num_frames`.
3. Probe one `anno.npz` to verify schema; raise `RuntimeError` with a clear message if keys are missing.

### `__len__`
Returns number of sequences. One sample per sequence per epoch.

### `__getitem__(idx)`
1. Pick `start_frame` such that `[start, start + num_frames * stride)` fits, where `stride` is sampled by `TemporalSubsampling.sample_stride()`.
2. Load `num_frames` RGB frames at the sampled indices, resize square to `img_size`, store as float32 `[0, 1]`.
3. Load matching depth, normals; slice `trajs_2d/trajs_3d/visibilities` along time.
4. Compute `aspect_ratio = (orig_H/max(orig_H, orig_W), orig_W/max(orig_H, orig_W))`.
5. Apply photometric `transform(video)`.
6. `q = sample_queries(...)` → `coords, t_src, t_tgt, t_cam, task_id, query_meta`.
7. `targets = build_targets(q, ...)`.
8. Return dict per §2.

### Edge cases
- Trajectories with no visible frame in window → filtered before becoming track queries.
- Pixels with invalid depth (NaN or `≤ 0`) → `mask_3d=0` for that query.
- Missing normals → `mask_normal=0`.
- Window-end out of bounds → resample `start_frame` (bounded retry); fall back to a different sequence if persistent.

---

## 5. Query sampling (`data/query_sampling.py`)

```python
def sample_queries(
    num_queries, num_frames, img_size,
    trajs_2d, trajs_3d, visibilities,
    depth, normals,
    task_mix, rng,
) -> dict:
    # returns: coords (N,2) in [0,1], t_src/t_tgt/t_cam (N,) int64,
    #          task_id (N,) int64, query_meta (dict of extras targets.py needs)
```

### Per-task emission rules
| task_id | task | Sampling |
|---|---|---|
| 0 | point-track | Pick a random trajectory whose visibility fraction in window ≥ `min_visible_frac`. `t_src = first frame in window where it's visible`. Emit `T` queries — `(u,v)` = traj position at `t_src` (normalized), `t_src` fixed, `t_tgt = t_cam = k` for `k ∈ [0, T)`. Repeat until quota. |
| 1 | depth | Random pixel `(u,v)`, random `t`. `t_src = t_tgt = t_cam = t`. |
| 2 | point-cloud | Random pixel `(u,v)`, random `t`. `t_src = t_tgt = t`. `t_cam = anchor` (anchor = middle frame, fixed per sample). |
| 3 | extrinsics | Random pixel `(u,v)`. `t_src = anchor`. `t_tgt = t_cam = random t`. |
| 4 | intrinsics | Random pixel `(u,v)`, random `t`. `t_src = t_tgt = t_cam = t`. (Identical tuple to task 1; separate `task_id` for diagnostics.) |

Each task receives `round(num_queries * task_mix[i])` queries; remainders go to point-track to keep the total at exactly `num_queries`.

### `query_meta`
Carries per-query info that `targets.py` needs but the model doesn't see — primarily `traj_idx` (or `-1` for non-track queries) and the un-normalized pixel coords used for depth/normal lookup.

### Determinism
`rng` is an `np.random.Generator` seeded per `__getitem__` from `(worker_seed, idx)`. Workers seed themselves in DataLoader's `worker_init_fn`; `PointOdyssey.__init__` provides a hook for this.

---

## 6. Target generation (`data/targets.py`)

```python
def build_targets(
    coords, t_src, t_tgt, t_cam, task_id, query_meta,
    trajs_2d, trajs_3d, visibilities,
    depth, normals,
    intrinsics, extrinsics,        # extrinsics may be None
    img_size,
) -> dict:
    # returns the loss-ready targets dict (see §2)
```

### Per-task `pos_3d` derivation (always in `t_cam` camera frame)
| task | Derivation |
|---|---|
| 0 (track) | `trajs_3d[t_tgt, traj_idx]` (world) → transform via `extrinsics[t_cam]`. |
| 1 (depth) | Back-project `(u_pix, v_pix)` at frame `t = t_tgt = t_cam` with `intrinsics[t]` and `depth[t, v_pix, u_pix]`. Already in that camera's frame. |
| 2 (point-cloud) | Back-project at `t_src` (camera-`t_src` coords) → transform via `extrinsics[t_cam] @ inv(extrinsics[t_src])`. |
| 3 (extrinsics) | Same math as task 2; `t_src` fixed at anchor, `t_cam` varies. |
| 4 (intrinsics) | Same as task 1. |

### Other targets
- `pos_2d`: project `pos_3d` into camera at `t_tgt` with `intrinsics[t_tgt]`, then divide by `img_size` to land in `[0, 1]`. Consistent with input `coords` convention.
- `visibility`:
  - task 0: `visibilities[t_tgt, traj_idx]`.
  - tasks 1–4: `depth[t_tgt, v_pix, u_pix] > 0` (used as both target and validity signal).
- `displacement`: only for task 0 (`pos_3d(t_tgt, t_cam) - pos_3d(t_src, t_cam)`); zero elsewhere with `mask_disp=0`.
- `normal`:
  - task 0: sample `normals[t_src]` at the trajectory's t_src pixel.
  - tasks 1–4: `normals[t_src, v_pix, u_pix]` directly.

### Masks
| Mask | Set to 1 when |
|---|---|
| `mask_3d` | `pos_3d` derivation succeeded (valid depth or visible trajectory). |
| `mask_2d` | Projected `pos_2d` ∈ `[0, 1]` AND visibility=1. |
| `mask_vis` | Visibility GT exists (always 1 in PointOdyssey when sourced from `visibilities`). |
| `mask_disp` | `task_id == 0` AND `pos_3d` valid at both `t_src` and `t_tgt`. |
| `mask_normal` | A normal could be sampled (in-bounds pixel + present file). |

### Open items (verify during implementation, not blocking design)
- **Coordinate handedness.** PointOdyssey's world-frame convention (axis order, +Z direction) vs. OpenCV's `+Z forward` camera convention. We'll write the back-projection math against the OpenCV convention and verify against a known sample (e.g., depth=1 at center pixel should land at `[0, 0, 1]` in camera coords).
- **`intrinsics.npy` shape.** May be `(3,3)` (constant) or `(T, 3, 3)` (per-frame). Code handles both.

---

## 7. Augmentation (`data/augmentations.py`)

```python
@dataclass
class AugmentationConfig:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05
    blur_prob: float = 0.1
    blur_sigma_max: float = 1.5
    temporal_stride_min: int = 1
    temporal_stride_max: int = 4

class VideoAugmentation:
    def __init__(self, cfg: AugmentationConfig): ...
    def __call__(self, video: np.ndarray) -> np.ndarray:
        # (T, H, W, 3) -> (T, H, W, 3); same color transform across all T frames
        ...

class TemporalSubsampling:
    def __init__(self, cfg: AugmentationConfig): ...
    def sample_stride(self, rng) -> int: ...
```

Key invariants:
- Color jitter parameters are sampled **once per `__call__`** and applied identically to all `T` frames. Per-frame resampling would destroy temporal coherence and the tracking signal.
- Gaussian blur similarly: single sigma per clip.
- No normalization here — the dataset's `__getitem__` handles `uint8 → float32 / 255.0`.
- `TemporalSubsampling` is stateless utility called by `PointOdyssey.__getitem__` to pick a stride before frame loading; it is **not** invoked by `VideoAugmentation`.

---

## 8. Stubs (`data/stubs.py`)

Each of `VideoDataset`, `KubricDataset`, `SintelDataset`, `ScanNetDataset` is a class whose `__init__` raises:
```python
raise NotImplementedError(
    f"{type(self).__name__} is not implemented yet. "
    f"Use --dataset pointodyssey. See docs/superpowers/specs/2026-05-25-data-module-design.md"
)
```
This keeps `from data import ...` working and gives a useful error if a config selects an unimplemented dataset.

---

## 9. Collate (`data/collate.py`)

```python
def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == 'targets':
            out[k] = {tk: torch.stack([b['targets'][tk] for b in batch])
                      for tk in batch[0]['targets']}
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
```
Default `torch.utils.data.dataloader.default_collate` mostly handles this but stumbles on nested dicts in some PyTorch versions; hand-rolling is two lines and obviously correct.

---

## 10. Testing

Single file: `tests/test_data.py`. Runs in <1s, no GPU, no real data.

### Fixture
`make_fake_pointodyssey(tmp_path)` builds a tiny on-disk dataset:
- 2 sequences, 64 frames each
- 64×64 JPEG frames with random pixel values
- Random `(64, 64)` float32 depth maps with positive values
- Random `(64, 64, 3)` normal maps, L2-normalized
- Synthetic `anno.npz` with 100 trajectories, plausible visibilities
- `intrinsics.npy` as `(64, 3, 3)` and `extrinsics.npy` as `(64, 4, 4)` with identity matrices

### Tests
1. **`test_pointodyssey_smoke`** — construct dataset on fixture, call `__getitem__(0)`, assert all required keys present with correct shapes and dtypes, `coords ∈ [0, 1]`, `t_* ∈ [0, num_frames)`.
2. **`test_collate_and_loss_compat`** — collate 2 samples, build a mock `predictions` dict matching the model's output spec, run `D4RTLoss(predictions, targets)`, assert the returned `loss` is a finite scalar. *This is the real safety net.*
3. **`test_task_mix_distribution`** — assert that across 1000 sampled queries the `task_id` distribution roughly matches `task_mix` (within a tolerance).
4. **`test_stubs_raise`** — each of the four stub classes raises `NotImplementedError` with the expected message.

---

## 11. Implementation order (informs the implementation plan)

Suggested sequence so the test surface grows with the code:
1. `augmentations.py` — small, no dependencies.
2. `base.py` + `collate.py` — small helpers.
3. `stubs.py` + `__init__.py` re-exports — unblocks `train.py` imports immediately.
4. Synthetic test fixture.
5. `query_sampling.py` with `test_task_mix_distribution`.
6. `targets.py` — built against the fake fixture.
7. `pointodyssey.py` — wires it all together.
8. `test_pointodyssey_smoke` + `test_collate_and_loss_compat`.
9. `train.py` patch (add `pointodyssey` choice).
10. Run `train.py` on a tiny subset (e.g. PointOdyssey sample, 1 sequence, 8 frames, 64 queries) to confirm one training step completes.

---

## 12. Out of scope (intentional)

- Boundary-aware sampling (CLAUDE.md mentions 30% near depth/motion edges).
- Geometric augmentation (flip/crop/rotate with target remapping).
- Caching/preprocessing of depth-edge maps offline.
- Full implementations of Kubric/Sintel/ScanNet/VideoDataset.
- `evaluate.py` integration beyond ensuring `split='val'`/`'test'` is accepted.
- TartanAir support.

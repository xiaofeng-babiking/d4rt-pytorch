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

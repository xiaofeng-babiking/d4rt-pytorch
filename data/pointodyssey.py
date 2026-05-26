"""PointOdysseyDataset for the real on-disk layout.

Per-sequence directory contains (singular names match the public release):
    rgb/00000.jpg, 00001.jpg, ...
    depth/00000.npy, 00001.npy, ...
    cam/00000.npz, 00001.npz, ...   each holds:
        intrinsics (3, 3) float32
        pose       (4, 4) float32   world->camera extrinsics

Optional (handled gracefully if absent):
    normals/00000.npy, ...
    anno.npz with keys trajs_2d (T,P,2), trajs_3d (T,P,3), visibilities (T,P)

When `anno.npz` is missing, point-track queries are disabled — `task_mix[0]`
is reset to 0 and its mass is redistributed to depth. When `normals/` is
missing, `mask_normal` is forced to 0 so the normal-loss term contributes
nothing. Both branches emit a one-time warning at construction time.
"""

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
REQUIRED_CAM_KEYS = {"intrinsics", "pose"}


def _load_cam_window(cam_dir: Path, frame_indices: np.ndarray):
    """Load per-frame cam/*.npz files at the requested indices.

    Returns:
        K: (T, 3, 3) float32 intrinsics (raw, NOT rescaled to img_size).
        E: (T, 4, 4) float32 world->camera extrinsics.
    """
    cam_files = sorted(f for f in cam_dir.iterdir() if f.suffix.lower() == ".npz")
    Ks, Es = [], []
    for t in frame_indices:
        with np.load(cam_files[int(t)]) as c:
            Ks.append(c["intrinsics"].astype(np.float32))
            Es.append(c["pose"].astype(np.float32))
    return np.stack(Ks, axis=0), np.stack(Es, axis=0)


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
            If `anno.npz` is missing, `task_mix[0]` is reset to 0 and its mass
            is added to task 1 (depth).
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
        rng_seed: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.num_queries = num_queries
        self.transform = transform

        if transform is not None and hasattr(transform, "cfg"):
            self._temporal = TemporalSubsampling(transform.cfg)
        else:
            self._temporal = TemporalSubsampling(AugmentationConfig())

        if rng_seed is None:
            self._base_seed = int(np.random.SeedSequence().entropy)
        else:
            self._base_seed = int(rng_seed)

        split_dir = self.data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"PointOdyssey split directory not found: {split_dir}"
            )

        sequences = []
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            rgb_dir = seq_dir / "rgb"
            if not rgb_dir.is_dir():
                continue
            frame_count = sum(1 for f in rgb_dir.iterdir()
                              if f.suffix.lower() in {".jpg", ".png"})
            if frame_count >= num_frames:
                sequences.append((seq_dir, frame_count))

        if not sequences:
            raise RuntimeError(
                f"No sequences in {split_dir} with rgb/ directory and "
                f">= {num_frames} frames."
            )

        probe_seq = sequences[0][0]
        if not (probe_seq / "depth").is_dir():
            raise RuntimeError(
                f"depth/ directory not found in first sequence: {probe_seq / 'depth'}"
            )
        cam_dir = probe_seq / "cam"
        if not cam_dir.is_dir():
            raise RuntimeError(
                f"cam/ directory not found in first sequence: {cam_dir}"
            )
        cam_files = sorted(f for f in cam_dir.iterdir() if f.suffix.lower() == ".npz")
        if not cam_files:
            raise RuntimeError(
                f"cam/ directory is empty in first sequence: {cam_dir}"
            )
        with np.load(cam_files[0]) as c:
            missing = REQUIRED_CAM_KEYS - set(c.files)
            if missing:
                raise RuntimeError(
                    f"{cam_files[0]} missing required keys: {missing} "
                    f"(expected: {REQUIRED_CAM_KEYS})"
                )

        self._has_anno = (probe_seq / "anno.npz").exists()
        self._has_normals = (probe_seq / "normals").is_dir()

        if self._has_anno:
            with np.load(probe_seq / "anno.npz") as a:
                anno_missing = REQUIRED_ANNO_KEYS - set(a.files)
                if anno_missing:
                    raise RuntimeError(
                        f"{probe_seq / 'anno.npz'} missing required keys: {anno_missing}"
                    )
        else:
            warnings.warn(
                f"No anno.npz in {probe_seq}; point-track queries disabled. "
                f"task_mix[0] mass redistributed to depth (task 1).",
                stacklevel=2,
            )

        if not self._has_normals:
            warnings.warn(
                f"No normals/ in {probe_seq}; mask_normal will be 0 for all "
                f"queries — normal-loss term contributes nothing.",
                stacklevel=2,
            )

        if not self._has_anno and task_mix[0] > 0:
            task_mix = (
                0.0,
                task_mix[1] + task_mix[0],
                task_mix[2], task_mix[3], task_mix[4],
            )
        self.task_mix = tuple(task_mix)

        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq_dir, frame_count = self.sequences[idx]

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        seed_seq = np.random.SeedSequence([self._base_seed, worker_id, idx])
        rng = np.random.default_rng(seed_seq)

        if self._has_anno:
            with np.load(seq_dir / "anno.npz") as a:
                trajs_2d_full = a["trajs_2d"].astype(np.float32)
                trajs_3d_full = a["trajs_3d"].astype(np.float32)
                visibilities_full = a["visibilities"].astype(np.float32)
            anno_T = trajs_2d_full.shape[0]
        else:
            trajs_2d_full = trajs_3d_full = visibilities_full = None
            anno_T = frame_count

        effective_frame_count = min(frame_count, anno_T)
        if effective_frame_count < self.num_frames:
            raise RuntimeError(
                f"Sequence {seq_dir} has only {effective_frame_count} aligned "
                f"frames (rgb={frame_count}, anno={anno_T}) but num_frames="
                f"{self.num_frames} was requested."
            )

        stride = self._temporal.sample_stride(rng)
        span = self.num_frames * stride
        while span > effective_frame_count and stride > 1:
            stride -= 1
            span = self.num_frames * stride
        max_start = max(0, effective_frame_count - span)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        frame_indices = np.arange(start, start + span, stride)[: self.num_frames]

        rgb_dir = seq_dir / "rgb"
        rgb_files = sorted(f for f in rgb_dir.iterdir()
                           if f.suffix.lower() in {".jpg", ".png"})
        raw = []
        for t in frame_indices:
            img = Image.open(rgb_files[int(t)]).convert("RGB")
            raw.append(np.asarray(img))
        raw = np.stack(raw, axis=0)
        orig_h, orig_w = raw.shape[1], raw.shape[2]
        aspect_ratio = compute_aspect_ratio(orig_h, orig_w)

        frames = resize_frames_square(raw, self.img_size)
        video = to_float32_normalized(frames)

        if self.transform is not None:
            video = self.transform(video)

        depth_files = sorted(f for f in (seq_dir / "depth").iterdir()
                             if f.suffix.lower() == ".npy")
        depth_stack = []
        for t in frame_indices:
            d = np.load(depth_files[int(t)])
            d_img = Image.fromarray(d.astype(np.float32), mode="F").resize(
                (self.img_size, self.img_size), Image.NEAREST)
            depth_stack.append(np.asarray(d_img, dtype=np.float32))
        depth = np.stack(depth_stack, axis=0)

        if self._has_normals:
            normal_files = sorted(f for f in (seq_dir / "normals").iterdir()
                                  if f.suffix.lower() == ".npy")
            normal_stack = []
            for t in frame_indices:
                n = np.load(normal_files[int(t)])
                n_img = np.stack([
                    np.asarray(Image.fromarray(n[..., c], mode="F").resize(
                        (self.img_size, self.img_size), Image.NEAREST),
                        dtype=np.float32)
                    for c in range(3)
                ], axis=-1)
                normal_stack.append(n_img)
            normals = np.stack(normal_stack, axis=0)
        else:
            normals = np.zeros((len(frame_indices), self.img_size, self.img_size, 3),
                               dtype=np.float32)

        sx = self.img_size / orig_w
        sy = self.img_size / orig_h
        if self._has_anno:
            trajs_2d = trajs_2d_full[frame_indices].copy()
            trajs_2d[..., 0] *= sx
            trajs_2d[..., 1] *= sy
            trajs_3d = trajs_3d_full[frame_indices].copy()
            visibilities = visibilities_full[frame_indices].copy()
        else:
            T = len(frame_indices)
            trajs_2d = np.zeros((T, 0, 2), dtype=np.float32)
            trajs_3d = np.zeros((T, 0, 3), dtype=np.float32)
            visibilities = np.zeros((T, 0), dtype=np.float32)

        K_window, E_window = _load_cam_window(seq_dir / "cam", frame_indices)
        K_window[:, 0, 0] *= sx
        K_window[:, 0, 2] *= sx
        K_window[:, 1, 1] *= sy
        K_window[:, 1, 2] *= sy

        q = sample_queries(
            num_queries=self.num_queries,
            num_frames=self.num_frames,
            img_size=self.img_size,
            trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=visibilities,
            depth=depth, normals=normals,
            task_mix=self.task_mix,
            rng=rng,
        )

        targets = build_targets(
            coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
            task_id=q["task_id"], query_meta=q["query_meta"],
            trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=visibilities,
            depth=depth, normals=normals,
            intrinsics=K_window, extrinsics=E_window, img_size=self.img_size,
        )

        if not self._has_normals:
            targets["mask_normal"] = torch.zeros_like(targets["mask_normal"])

        return {
            "video": torch.from_numpy(video),
            "coords": torch.from_numpy(q["coords"]).float(),
            "t_src": torch.from_numpy(q["t_src"]).long(),
            "t_tgt": torch.from_numpy(q["t_tgt"]).long(),
            "t_cam": torch.from_numpy(q["t_cam"]).long(),
            "aspect_ratio": torch.from_numpy(aspect_ratio).float(),
            "targets": targets,
        }

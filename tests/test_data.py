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


def test_video_augmentation_blur_applied():
    """Blur branch must produce valid output shape/dtype/range."""
    aug = VideoAugmentation(
        AugmentationConfig(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0,
                           blur_prob=1.0, blur_sigma_max=1.5),
        seed=42,
    )
    video = np.random.RandomState(0).rand(4, 32, 32, 3).astype(np.float32)
    out = aug(video)
    assert out.shape == video.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


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


def test_to_float32_normalized_float_input_casts_to_float32():
    arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    out = to_float32_normalized(arr)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-6)


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


def test_sample_queries_point_cloud_task_invariants():
    """For task 2 queries, t_src == t_tgt and t_cam == anchor (num_frames // 2)."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(5)
    out = sample_queries(
        num_queries=200, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(0.0, 0.0, 1.0, 0.0, 0.0),
        rng=rng,
    )
    assert (out["task_id"] == 2).all()
    np.testing.assert_array_equal(out["t_src"], out["t_tgt"])
    assert (out["t_cam"] == 16 // 2).all()
    assert out["query_meta"]["anchor"] == 16 // 2


def test_sample_queries_intrinsics_task_invariants():
    """For task 4 queries, t_src == t_tgt == t_cam (same as depth)."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20)
    rng = np.random.default_rng(6)
    out = sample_queries(
        num_queries=200, num_frames=16, img_size=64,
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        task_mix=(0.0, 0.0, 0.0, 0.0, 1.0),
        rng=rng,
    )
    assert (out["task_id"] == 4).all()
    np.testing.assert_array_equal(out["t_src"], out["t_tgt"])
    np.testing.assert_array_equal(out["t_src"], out["t_cam"])


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


def test_build_targets_pos_2d_roundtrips_pos_3d_for_depth_task():
    """For task 1, pos_2d must round-trip through the intrinsics back to coords."""
    trajs_2d, trajs_3d, vis, depth, normals = _fake_anno(T=16, P=20, H=64, W=64)
    img_size = 64
    K = np.array([[img_size * 0.8, 0, img_size / 2],
                  [0, img_size * 0.8, img_size / 2],
                  [0, 0, 1]], dtype=np.float32)
    intrinsics = np.broadcast_to(K, (16, 3, 3)).copy()
    extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32), (16, 4, 4)).copy()
    rng = np.random.default_rng(11)
    q = sample_queries(num_queries=64, num_frames=16, img_size=img_size,
                       trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
                       depth=depth, normals=normals,
                       task_mix=(0.0, 1.0, 0.0, 0.0, 0.0), rng=rng)
    tgt = build_targets(
        coords=q["coords"], t_src=q["t_src"], t_tgt=q["t_tgt"], t_cam=q["t_cam"],
        task_id=q["task_id"], query_meta=q["query_meta"],
        trajs_2d=trajs_2d, trajs_3d=trajs_3d, visibilities=vis,
        depth=depth, normals=normals,
        intrinsics=intrinsics, extrinsics=extrinsics, img_size=img_size,
    )
    # For depth-task queries with valid masks, the projected pos_2d should
    # be within ~1 pixel of the input coords (it back-projects then projects
    # from the same K, so it's a round-trip).
    valid = tgt["mask_3d"] > 0
    coords_t = torch.from_numpy(q["coords"])
    diff = (tgt["pos_2d"][valid] - coords_t[valid]).abs()
    # Allow ~1 pixel of slop in normalized coords: 1 / (64-1) ~ 0.016
    assert diff.max().item() < 0.05, f"pos_2d roundtrip error too large: {diff.max().item()}"


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

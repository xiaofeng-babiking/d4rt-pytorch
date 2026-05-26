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

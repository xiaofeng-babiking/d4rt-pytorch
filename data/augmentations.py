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

    def __init__(self, cfg: Optional[AugmentationConfig] = None,
                 seed: Optional[int] = None):
        self.cfg = cfg or AugmentationConfig()
        self._rng = np.random.default_rng(seed)

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

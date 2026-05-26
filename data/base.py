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
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        out[i] = np.asarray(img)
    return out


def compute_aspect_ratio(orig_h: int, orig_w: int) -> np.ndarray:
    """Return (h/max, w/max) so encoders can distinguish landscape vs portrait.

    The longer side becomes 1.0; the shorter side is the ratio.
    """
    m = max(orig_h, orig_w)
    return np.array([orig_h / m, orig_w / m], dtype=np.float32)


def to_float32_normalized(arr: np.ndarray) -> np.ndarray:
    """Convert to float32 in [0, 1].

    - uint8 input: scaled by 1/255 to [0, 1].
    - Any other input: cast to float32; values are assumed already normalized.
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr.astype(np.float32)

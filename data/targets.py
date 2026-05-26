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


def _per_query_E(extrinsics: Optional[np.ndarray], t: np.ndarray):
    if extrinsics is None:
        return np.broadcast_to(np.eye(4, dtype=np.float32),
                                (t.shape[0], 4, 4)).copy()
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

        # 2D projection in camera-t_cam frame (cam_tgt is already in t_cam coords).
        K_cam = _per_query_K(intrinsics, tcam)
        pix = _project(cam_tgt, K_cam) / (img_size - 1)
        pos_2d[idxs] = pix.astype(np.float32)

        # Visibility from annotations
        vis_vals = visibilities[ttgt, p].astype(np.float32)
        visibility[idxs] = vis_vals
        mask_vis[idxs] = 1.0

        # Mask 3D where Z is positive AND values are finite (guard NaN/Inf).
        valid_3d = (cam_tgt[:, 2] > 0) & np.isfinite(cam_tgt).all(axis=1)
        valid_src = (cam_src[:, 2] > 0) & np.isfinite(cam_src).all(axis=1)
        mask_3d[idxs] = valid_3d.astype(np.float32)
        # Image-bounds check for pos_2d (spec §6: mask_2d requires pos_2d in [0,1]).
        in_bounds = (pix[:, 0] >= 0) & (pix[:, 0] <= 1) & (pix[:, 1] >= 0) & (pix[:, 1] <= 1)
        mask_2d[idxs] = (valid_3d & (vis_vals > 0.5) & in_bounds).astype(np.float32)
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
        in_bounds = (pix[:, 0] >= 0) & (pix[:, 0] <= 1) & (pix[:, 1] >= 0) & (pix[:, 1] <= 1)
        mask_2d[idxs] = (valid & in_bounds).astype(np.float32)
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
        in_bounds = (pix[:, 0] >= 0) & (pix[:, 0] <= 1) & (pix[:, 1] >= 0) & (pix[:, 1] <= 1)
        mask_2d[idxs] = (valid & in_bounds).astype(np.float32)
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

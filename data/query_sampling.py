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
    """Split `total` queries by `mix` ratios via floor-rounding.

    Per spec §5: any remainder from floor-rounding is assigned to task 0
    (point-track). This is intentional — point-track has the strongest
    training signal in PointOdyssey, so biasing toward it under rounding
    is a feature, not a bug. Do not change to largest-remainder.
    """
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

    Note: `img_size` is a scalar because the upstream dataset (PointOdyssey)
    resizes all frames to a square and rescales `trajs_2d` to match before
    this function is called. If a future caller passes non-square frames,
    this normalization will need to take separate height/width.
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
    """task 4: same query tuple as depth (task 1).

    The (u, v, t_src, t_tgt, t_cam) tuples are statistically identical to
    task 1. We emit them with a separate `task_id=4` purely so dataset-side
    diagnostics and logging can distinguish "intrinsics-flavored" queries
    from depth queries — the model itself does NOT see task_id and treats
    both identically. Per spec §5, this duplication is intentional.
    """
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

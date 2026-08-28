"""Tiled heightfield generation and bilinear height sampling for rough terrain."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Must match scene_rough.xml <hfield size nrow ncol>.
HFIELD_NROW = 128
HFIELD_NCOL = 128
HFIELD_HALF_X = 16.0
HFIELD_HALF_Y = 16.0
# Max stored elevation (m). Large enough for ~8–12° ramps over an 8 m tile.
HFIELD_ELEVATION_Z = 1.5
HFIELD_BASE_Z = 0.1
HFIELD_SIZE = (HFIELD_HALF_X, HFIELD_HALF_Y, HFIELD_ELEVATION_Z, HFIELD_BASE_Z)

N_TILES = 4
TILE_METERS = 8.0
DEFAULT_SEED = 0
SPAWN_PAD_M = 1.6
MAP_EDGE_MARGIN = 0.5

# Peak amplitudes at terrain_scale=1.0 (meters).
ROUGH_AMP = 0.10
STAIR_STEP = 0.07
STAIR_COUNT = 8
SLOPE_DEG = 10.0
MIXED_BOX_H = 0.08

# Row-major from min-y to max-y, min-x to max-x.
TILE_LAYOUT: Tuple[Tuple[str, ...], ...] = (
    ("flat", "rough", "slope_x", "stairs"),
    ("rough", "mixed", "slope_y", "rough"),
    ("stairs", "slope_x", "rough", "flat"),
    ("mixed", "rough", "stairs", "slope_y"),
)


def _upsample_bilinear(grid: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    gh, gw = grid.shape
    ys = np.linspace(0.0, gh - 1, nrow, dtype=np.float32)
    xs = np.linspace(0.0, gw - 1, ncol, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    sy = (ys - y0).astype(np.float32)
    sx = (xs - x0).astype(np.float32)
    g00 = grid[y0][:, x0]
    g10 = grid[y0][:, x1]
    g01 = grid[y1][:, x0]
    g11 = grid[y1][:, x1]
    top = g00 * (1.0 - sx) + g10 * sx
    bot = g01 * (1.0 - sx) + g11 * sx
    return (top * (1.0 - sy)[:, None] + bot * sy[:, None]).astype(np.float32)


def _fractal_noise(
    rng: np.random.Generator,
    nrow: int,
    ncol: int,
    *,
    base_res: int = 6,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    acc = np.zeros((nrow, ncol), dtype=np.float32)
    amp = 1.0
    total = 0.0
    res = base_res
    for _ in range(octaves):
        grid = rng.uniform(0.0, 1.0, size=(res + 1, res + 1)).astype(np.float32)
        acc += amp * _upsample_bilinear(grid, nrow, ncol)
        total += amp
        amp *= persistence
        res = min(res * 2, max(nrow, ncol))
    acc = acc / max(total, 1e-6)
    acc -= acc.mean()
    peak = np.max(np.abs(acc))
    if peak > 1e-6:
        acc = acc / peak
    return acc.astype(np.float32)


def _tile_slices(
    nrow: int, ncol: int, tile_i: int, tile_j: int
) -> Tuple[slice, slice]:
    r0 = int(round(tile_i * nrow / N_TILES))
    r1 = int(round((tile_i + 1) * nrow / N_TILES))
    c0 = int(round(tile_j * ncol / N_TILES))
    c1 = int(round((tile_j + 1) * ncol / N_TILES))
    return slice(r0, r1), slice(c0, c1)


def _local_lin(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def _fill_tile(
    heights: np.ndarray,
    rng: np.random.Generator,
    kind: str,
    scale: float,
    tile_i: int,
    tile_j: int,
) -> None:
    rs, cs = _tile_slices(*heights.shape, tile_i, tile_j)
    block = heights[rs, cs]
    nr, nc = block.shape
    yy = _local_lin(nr)[:, None]
    xx = _local_lin(nc)[None, :]
    s = float(np.clip(scale, 0.0, 1.0))

    if kind == "flat":
        block[:, :] = 0.0
    elif kind == "rough":
        noise = _fractal_noise(rng, nr, nc, base_res=5, octaves=4)
        block[:, :] = np.maximum(0.0, 0.5 * ROUGH_AMP * s * (noise + 1.0))
    elif kind == "slope_x":
        rise = float(np.tan(np.deg2rad(SLOPE_DEG)) * TILE_METERS) * s
        noise = _fractal_noise(rng, nr, nc, base_res=4, octaves=3) * (0.02 * s)
        block[:, :] = np.clip(xx * rise + noise, 0.0, None)
    elif kind == "slope_y":
        rise = float(np.tan(np.deg2rad(SLOPE_DEG)) * TILE_METERS) * s
        noise = _fractal_noise(rng, nr, nc, base_res=4, octaves=3) * (0.02 * s)
        block[:, :] = np.clip(yy * rise + noise, 0.0, None)
    elif kind == "stairs":
        step_h = STAIR_STEP * s
        idx = np.floor(xx * STAIR_COUNT)
        idx = np.clip(idx, 0.0, STAIR_COUNT - 1.0)
        block[:, :] = idx * step_h
    elif kind == "mixed":
        noise = _fractal_noise(rng, nr, nc, base_res=5, octaves=4)
        block[:, :] = np.maximum(0.0, 0.35 * ROUGH_AMP * s * (noise + 1.0))
        for _ in range(2):
            r0 = int(rng.integers(max(nr // 8, 1), max(nr - nr // 4, 2)))
            c0 = int(rng.integers(max(nc // 8, 1), max(nc - nc // 4, 2)))
            rh = max(int(nr * 0.18), 2)
            cw = max(int(nc * 0.18), 2)
            r1 = min(nr, r0 + rh)
            c1 = min(nc, c0 + cw)
            block[r0:r1, c0:c1] = np.maximum(block[r0:r1, c0:c1], MIXED_BOX_H * s)
    else:
        raise ValueError(f"Unknown tile kind: {kind}")
    heights[rs, cs] = block


def _flatten_spawn_pads(
    heights: np.ndarray, half_x: float, half_y: float
) -> None:
    """Flatten a small pad at each tile center so spawn feet are not buried."""
    nrow, ncol = heights.shape
    pad_r = max(int(round(0.5 * SPAWN_PAD_M / (2.0 * half_y) * (nrow - 1))), 2)
    pad_c = max(int(round(0.5 * SPAWN_PAD_M / (2.0 * half_x) * (ncol - 1))), 2)
    for i in range(N_TILES):
        for j in range(N_TILES):
            cx, cy = tile_center(i, j, half_x, half_y)
            row, col = world_to_index(cx, cy, nrow, ncol, half_x, half_y)
            r0 = max(0, row - pad_r)
            r1 = min(nrow, row + pad_r + 1)
            c0 = max(0, col - pad_c)
            c1 = min(ncol, col + pad_c + 1)
            pad_h = float(heights[row, col])
            heights[r0:r1, c0:c1] = pad_h


def tile_center(
    tile_i: int,
    tile_j: int,
    half_x: float = HFIELD_HALF_X,
    half_y: float = HFIELD_HALF_Y,
) -> Tuple[float, float]:
    x0 = -half_x + float(tile_j) * TILE_METERS
    y0 = -half_y + float(tile_i) * TILE_METERS
    return x0 + 0.5 * TILE_METERS, y0 + 0.5 * TILE_METERS


def world_to_index(
    x: float,
    y: float,
    nrow: int,
    ncol: int,
    half_x: float,
    half_y: float,
) -> Tuple[int, int]:
    col = (x / half_x + 1.0) * 0.5 * (ncol - 1)
    row = (y / half_y + 1.0) * 0.5 * (nrow - 1)
    return int(np.clip(np.round(row), 0, nrow - 1)), int(
        np.clip(np.round(col), 0, ncol - 1)
    )


def generate_tiled_heights(
    nrow: int = HFIELD_NROW,
    ncol: int = HFIELD_NCOL,
    *,
    half_x: float = HFIELD_HALF_X,
    half_y: float = HFIELD_HALF_Y,
    scale: float = 1.0,
    seed: int = DEFAULT_SEED,
    layout: Sequence[Sequence[str]] = TILE_LAYOUT,
) -> np.ndarray:
    """Return (nrow, ncol) elevation in meters, row 0 = min y."""
    rng = np.random.default_rng(int(seed))
    heights = np.zeros((nrow, ncol), dtype=np.float32)
    for i, row in enumerate(layout):
        for j, kind in enumerate(row):
            _fill_tile(heights, rng, kind, scale, i, j)
    _flatten_spawn_pads(heights, half_x, half_y)
    return heights


def heights_to_hfield_data(heights_m: np.ndarray, elevation_z: float) -> np.ndarray:
    return np.clip(heights_m / max(float(elevation_z), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def generate_hfield_data(
    nrow: int = HFIELD_NROW,
    ncol: int = HFIELD_NCOL,
    *,
    half_x: float = HFIELD_HALF_X,
    half_y: float = HFIELD_HALF_Y,
    elevation_z: float = HFIELD_ELEVATION_Z,
    scale: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Normalized [0, 1] row-major elevation for ``model.hfield_data``."""
    heights = generate_tiled_heights(
        nrow,
        ncol,
        half_x=half_x,
        half_y=half_y,
        scale=scale,
        seed=seed,
    )
    return heights_to_hfield_data(heights, elevation_z)


def sample_height(
    elev_01: np.ndarray,
    x: float,
    y: float,
    *,
    nrow: int,
    ncol: int,
    half_x: float,
    half_y: float,
    elevation_z: float,
) -> float:
    """Bilinear sample of hfield elevation (meters) at world (x, y)."""
    u = (float(x) / half_x + 1.0) * 0.5 * (ncol - 1)
    v = (float(y) / half_y + 1.0) * 0.5 * (nrow - 1)
    u = float(np.clip(u, 0.0, ncol - 1.0001))
    v = float(np.clip(v, 0.0, nrow - 1.0001))
    u0 = int(np.floor(u))
    v0 = int(np.floor(v))
    u1 = min(u0 + 1, ncol - 1)
    v1 = min(v0 + 1, nrow - 1)
    su = u - u0
    sv = v - v0
    z00 = float(elev_01[v0, u0])
    z10 = float(elev_01[v0, u1])
    z01 = float(elev_01[v1, u0])
    z11 = float(elev_01[v1, u1])
    z = (
        (1.0 - su) * (1.0 - sv) * z00
        + su * (1.0 - sv) * z10
        + (1.0 - su) * sv * z01
        + su * sv * z11
    )
    return float(z * elevation_z)


def out_of_map(
    x: float,
    y: float,
    half_x: float = HFIELD_HALF_X,
    half_y: float = HFIELD_HALF_Y,
    margin: float = MAP_EDGE_MARGIN,
) -> bool:
    return abs(x) > half_x - margin or abs(y) > half_y - margin

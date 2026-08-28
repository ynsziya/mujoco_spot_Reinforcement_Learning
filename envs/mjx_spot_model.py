"""Shared Spot MjModel / MJX model loader with GPU-friendly patches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco
from mujoco import mjx
import numpy as np

from envs.terrain import (
    DEFAULT_SEED,
    HFIELD_ELEVATION_Z,
    HFIELD_HALF_X,
    HFIELD_HALF_Y,
    generate_hfield_data,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "scene.xml"
ROUGH_SCENE = ROOT / "scene_rough.xml"

N_LEG = 12
LEG_NAMES = ("fl", "fr", "rl", "rr")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEG_NAMES)
FOOT_SITES = tuple(f"{leg}_foot_site" for leg in LEG_NAMES)
LEG_JOINT_NAMES = (
    "fl_hip_roll_joint",
    "fl_hip_yaw_joint",
    "fl_knee_joint",
    "fr_hip_roll_joint",
    "fr_hip_yaw_joint",
    "fr_knee_joint",
    "rl_hip_roll_joint",
    "rl_hip_yaw_joint",
    "rl_knee_joint",
    "rr_hip_roll_joint",
    "rr_hip_yaw_joint",
    "rr_knee_joint",
)


@dataclass(frozen=True)
class SpotIds:
    home_key_id: int
    base_body_id: int
    floor_geom_id: int
    foot_geom_ids: np.ndarray
    foot_site_ids: np.ndarray
    leg_qposadr: np.ndarray
    leg_dofadr: np.ndarray
    leg_jnt_range: np.ndarray
    default_pose: np.ndarray
    home_qpos: np.ndarray
    home_qvel: np.ndarray
    home_ctrl: np.ndarray
    target_height: float
    frame_skip: int
    dt: float
    nq: int
    nv: int
    nu: int
    has_hfield: bool
    hfield_nrow: int
    hfield_ncol: int
    hfield_half_x: float
    hfield_half_y: float
    hfield_elevation_z: float
    hfield_elev: np.ndarray


def apply_mjx_patches(
    model: mujoco.MjModel,
    *,
    timestep: float = 0.004,
    iterations: int = 4,
    ls_iterations: int = 5,
    foot_condim: int = 3,
    armature: float | None = 0.02,
) -> mujoco.MjModel:
    """Mutate *model* in-place for fast, stable MJX simulation."""
    model.opt.timestep = float(timestep)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1.0
    model.opt.iterations = int(iterations)
    model.opt.ls_iterations = int(ls_iterations)
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON

    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is not None and name.endswith("_foot"):
            model.geom_condim[i] = int(foot_condim)

    if armature is not None:
        model.dof_armature[:] = np.maximum(model.dof_armature, float(armature))

    return model


def apply_terrain_hfield(
    model: mujoco.MjModel,
    *,
    scale: float = 1.0,
    seed: int = DEFAULT_SEED,
) -> mujoco.MjModel:
    """Fill the first heightfield with a tiled rough-terrain map."""
    if model.nhfield < 1:
        return model
    adr = int(model.hfield_adr[0])
    nrow = int(model.hfield_nrow[0])
    ncol = int(model.hfield_ncol[0])
    size = np.asarray(model.hfield_size[0], dtype=np.float64)
    elev = generate_hfield_data(
        nrow,
        ncol,
        half_x=float(size[0]),
        half_y=float(size[1]),
        elevation_z=float(size[2]),
        scale=scale,
        seed=seed,
    )
    model.hfield_data[adr : adr + nrow * ncol] = elev.reshape(-1)
    return model


def _hfield_fields(model: mujoco.MjModel) -> dict:
    if model.nhfield < 1:
        return dict(
            has_hfield=False,
            hfield_nrow=0,
            hfield_ncol=0,
            hfield_half_x=HFIELD_HALF_X,
            hfield_half_y=HFIELD_HALF_Y,
            hfield_elevation_z=HFIELD_ELEVATION_Z,
            hfield_elev=np.zeros((1, 1), dtype=np.float32),
        )
    nrow = int(model.hfield_nrow[0])
    ncol = int(model.hfield_ncol[0])
    adr = int(model.hfield_adr[0])
    size = np.asarray(model.hfield_size[0], dtype=np.float64)
    elev = np.asarray(
        model.hfield_data[adr : adr + nrow * ncol], dtype=np.float32
    ).reshape(nrow, ncol)
    return dict(
        has_hfield=True,
        hfield_nrow=nrow,
        hfield_ncol=ncol,
        hfield_half_x=float(size[0]),
        hfield_half_y=float(size[1]),
        hfield_elevation_z=float(size[2]),
        hfield_elev=elev,
    )


def resolve_ids(model: mujoco.MjModel, frame_skip: int = 5) -> SpotIds:
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id < 0:
        raise RuntimeError("Keyframe 'home' not found.")

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise RuntimeError("Body 'base_link' not found.")

    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id < 0:
        raise RuntimeError("Geom 'floor' not found.")

    foot_geom_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in FOOT_GEOMS
        ],
        dtype=np.int32,
    )
    foot_site_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in FOOT_SITES
        ],
        dtype=np.int32,
    )
    if np.any(foot_geom_ids < 0) or np.any(foot_site_ids < 0):
        raise RuntimeError("Foot geom/site names missing from model.")

    leg_jnt_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in LEG_JOINT_NAMES
        ],
        dtype=np.int32,
    )
    if np.any(leg_jnt_ids < 0):
        raise RuntimeError("Leg joint names missing from model.")
    leg_qposadr = model.jnt_qposadr[leg_jnt_ids].astype(np.int32)
    leg_dofadr = model.jnt_dofadr[leg_jnt_ids].astype(np.int32)
    leg_jnt_range = np.asarray(model.jnt_range[leg_jnt_ids], dtype=np.float32)

    default_pose = np.asarray(model.key_ctrl[home_key_id, :N_LEG], dtype=np.float32)
    home_qpos = np.asarray(model.key_qpos[home_key_id], dtype=np.float32)
    home_qvel = np.zeros(model.nv, dtype=np.float32)
    home_ctrl = np.asarray(model.key_ctrl[home_key_id], dtype=np.float32)
    target_height = float(home_qpos[2])
    dt = float(model.opt.timestep) * int(frame_skip)

    return SpotIds(
        home_key_id=int(home_key_id),
        base_body_id=int(base_body_id),
        floor_geom_id=int(floor_geom_id),
        foot_geom_ids=foot_geom_ids,
        foot_site_ids=foot_site_ids,
        leg_qposadr=leg_qposadr,
        leg_dofadr=leg_dofadr,
        leg_jnt_range=leg_jnt_range,
        default_pose=default_pose,
        home_qpos=home_qpos,
        home_qvel=home_qvel,
        home_ctrl=home_ctrl,
        target_height=target_height,
        frame_skip=int(frame_skip),
        dt=dt,
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        **_hfield_fields(model),
    )


def load_spot_mj_model(
    scene_path: str | Path | None = None,
    *,
    fast: bool = True,
    timestep: float = 0.004,
    frame_skip: int = 5,
    iterations: int = 4,
    ls_iterations: int = 5,
    terrain_scale: float = 1.0,
    terrain_seed: int = DEFAULT_SEED,
) -> Tuple[mujoco.MjModel, SpotIds]:
    path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE
    model = mujoco.MjModel.from_xml_path(str(path))
    if fast:
        apply_mjx_patches(
            model,
            timestep=timestep,
            iterations=iterations,
            ls_iterations=ls_iterations,
        )
    else:
        model.opt.timestep = float(timestep)
    apply_terrain_hfield(model, scale=terrain_scale, seed=terrain_seed)
    ids = resolve_ids(model, frame_skip=frame_skip)
    return model, ids


def load_spot_mjx(
    scene_path: str | Path | None = None,
    *,
    fast: bool = True,
    timestep: float = 0.004,
    frame_skip: int = 5,
    iterations: int = 4,
    ls_iterations: int = 5,
    terrain_scale: float = 1.0,
    terrain_seed: int = DEFAULT_SEED,
) -> Tuple[mujoco.MjModel, mjx.Model, SpotIds]:
    mj_model, ids = load_spot_mj_model(
        scene_path,
        fast=fast,
        timestep=timestep,
        frame_skip=frame_skip,
        iterations=iterations,
        ls_iterations=ls_iterations,
        terrain_scale=terrain_scale,
        terrain_seed=terrain_seed,
    )
    mjx_model = mjx.put_model(mj_model)
    return mj_model, mjx_model, ids

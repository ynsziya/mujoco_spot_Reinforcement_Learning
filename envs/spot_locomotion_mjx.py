"""Brax Env for Spot locomotion on MuJoCo MJX (walk + run)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from brax.envs.base import Env, State
from flax import struct
import jax
import jax.numpy as jnp
from mujoco import mjx

from envs.mjx_spot_model import N_LEG, ROUGH_SCENE, load_spot_mjx
from envs.terrain import (
    DEFAULT_SEED,
    MAP_EDGE_MARGIN,
    N_TILES,
    TILE_METERS,
)

OBS_DIM = 60
STACK_FRAMES = 3
STACKED_OBS_DIM = OBS_DIM * STACK_FRAMES
ACTION_DIM = N_LEG


@struct.dataclass
class EnvConfig:
    """Locomotion hyperparameters (immutable flax struct)."""

    stage: str = "walk"
    max_episode_steps: int = 1000
    action_filter: float = 0.55
    action_scale_hx: float = 0.25
    action_scale_hy: float = 0.45
    action_scale_knee: float = 0.45
    fall_height: float = 0.30
    max_tilt: float = 0.70
    vx_min: float = 0.0
    vx_max: float = 1.0
    vy_max: float = 0.35
    yaw_max: float = 0.6
    zero_cmd_prob: float = 0.10
    vy_zero_prob: float = 0.55
    yaw_zero_prob: float = 0.55
    heading_stiffness: float = 1.25
    cmd_resample_steps: int = 250
    gait_period_slow: float = 0.62
    gait_period_fast: float = 0.28
    duty_slow: float = 0.60
    duty_fast: float = 0.40
    obs_noise_scale: float = 0.02
    push_interval_steps: int = 100
    push_vel_xy: float = 0.5
    tracking_lin_sigma: float = 0.25
    tracking_ang_sigma: float = 0.25
    tracking_lin_w: float = 1.5
    tracking_ang_w: float = 1.4
    heading_w: float = 0.8
    heading_sigma: float = 0.25
    upright_w: float = 0.5
    height_w: float = 0.5
    air_time_w: float = 0.15
    clearance_w: float = 0.1
    phase_w: float = 0.3
    alive_w: float = 0.5
    vertical_vel_w: float = 0.2
    ang_xy_w: float = 0.05
    action_rate_w: float = 0.01
    action_w: float = 0.001
    joint_vel_w: float = 0.001
    joint_acc_w: float = 2.5e-7
    slip_w: float = 0.1
    limit_w: float = 0.1
    terminate_w: float = 10.0


def config_for_stage(stage: str, **overrides: Any) -> EnvConfig:
    stage = stage.lower()
    if stage == "walk":
        base = dict(stage="walk", vx_min=0.0, vx_max=1.0, vy_max=0.35, yaw_max=0.6)
    elif stage == "run":
        # 5 m/s ≈ 5× walk. Faster gaits + larger leg travel; use --vx-max to push further.
        base = dict(
            stage="run",
            vx_min=-0.3,
            vx_max=5.0,
            vy_max=0.6,
            yaw_max=1.2,
            gait_period_slow=0.50,
            gait_period_fast=0.16,
            duty_slow=0.55,
            duty_fast=0.32,
            action_filter=0.40,
            action_scale_hx=0.32,
            action_scale_hy=0.58,
            action_scale_knee=0.58,
            tracking_lin_sigma=0.35,
            air_time_w=0.25,
            clearance_w=0.15,
            fall_height=0.28,
        )
    elif stage == "rough":
        base = dict(
            stage="rough",
            vx_min=0.0,
            vx_max=0.8,
            vy_max=0.25,
            yaw_max=0.5,
            gait_period_slow=0.62,
            gait_period_fast=0.40,
            duty_slow=0.65,
            duty_fast=0.50,
            fall_height=0.28,
            clearance_w=0.2,
            air_time_w=0.1,
            vertical_vel_w=0.05,
            height_w=0.4,
            tracking_lin_sigma=0.30,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")
    base.update(overrides)
    return EnvConfig(**base)


class SpotLocomotionEnv(Env):
    """MJX Spot locomotion with velocity commands (walk/run)."""

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        *,
        scene_path: Optional[str] = None,
        fast: bool = True,
        timestep: float = 0.004,
        frame_skip: int = 5,
        iterations: int = 4,
        ls_iterations: int = 5,
        terrain_scale: float = 1.0,
        terrain_seed: int = DEFAULT_SEED,
    ):
        self.config = config or config_for_stage("walk")
        if scene_path is None and self.config.stage == "rough":
            scene_path = str(ROUGH_SCENE)
        self.mj_model, self.mjx_model, self.ids = load_spot_mjx(
            scene_path,
            fast=fast,
            timestep=timestep,
            frame_skip=frame_skip,
            iterations=iterations,
            ls_iterations=ls_iterations,
            terrain_scale=terrain_scale,
            terrain_seed=terrain_seed,
        )
        self.sys = self.mjx_model

        self._frame_skip = int(self.ids.frame_skip)
        self._dt = float(self.ids.dt)
        self._default_pose = jnp.asarray(self.ids.default_pose, dtype=jnp.float32)
        self._home_qpos = jnp.asarray(self.ids.home_qpos, dtype=jnp.float32)
        self._home_qvel = jnp.asarray(self.ids.home_qvel, dtype=jnp.float32)
        self._home_ctrl = jnp.asarray(self.ids.home_ctrl, dtype=jnp.float32)
        cfg = self.config
        self._action_scale = jnp.tile(
            jnp.array(
                [cfg.action_scale_hx, cfg.action_scale_hy, cfg.action_scale_knee],
                dtype=jnp.float32,
            ),
            4,
        )
        self._gait_offsets = jnp.array([0.0, 0.5, 0.5, 0.0], dtype=jnp.float32)
        self._foot_geom_ids = jnp.asarray(self.ids.foot_geom_ids, dtype=jnp.int32)
        self._foot_site_ids = jnp.asarray(self.ids.foot_site_ids, dtype=jnp.int32)
        self._leg_qposadr = jnp.asarray(self.ids.leg_qposadr, dtype=jnp.int32)
        self._leg_dofadr = jnp.asarray(self.ids.leg_dofadr, dtype=jnp.int32)
        self._base_body_id = int(self.ids.base_body_id)
        self._floor_geom_id = int(self.ids.floor_geom_id)
        self._target_height = float(self.ids.target_height)
        self._joint_low = jnp.asarray(self.ids.leg_jnt_range[:, 0], dtype=jnp.float32)
        self._joint_high = jnp.asarray(self.ids.leg_jnt_range[:, 1], dtype=jnp.float32)
        self._has_hfield = bool(self.ids.has_hfield)
        self._hfield_nrow = int(self.ids.hfield_nrow)
        self._hfield_ncol = int(self.ids.hfield_ncol)
        self._hfield_half_x = float(self.ids.hfield_half_x)
        self._hfield_half_y = float(self.ids.hfield_half_y)
        self._hfield_elevation_z = float(self.ids.hfield_elevation_z)
        self._hfield_elev = jnp.asarray(self.ids.hfield_elev, dtype=jnp.float32)

    def _leg_qpos(self, data: mjx.Data) -> jax.Array:
        return data.qpos[self._leg_qposadr]

    def _leg_qvel(self, data: mjx.Data) -> jax.Array:
        return data.qvel[self._leg_dofadr]

    @property
    def observation_size(self) -> int:
        return STACKED_OBS_DIM

    @property
    def action_size(self) -> int:
        return ACTION_DIM

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def dt(self) -> float:
        return self._dt

    def _yaw_from_quat(self, quat_wxyz: jax.Array) -> jax.Array:
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return jnp.arctan2(siny_cosp, cosy_cosp)

    def _wrap_to_pi(self, angle: jax.Array) -> jax.Array:
        return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    def _sample_velocity_command(
        self, rng: jax.Array, current_yaw: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Sample (vx, vy) and a heading target. Yaw rate is derived later."""
        cfg = self.config
        rng, k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 7)
        vx = jax.random.uniform(k1, (), minval=cfg.vx_min, maxval=cfg.vx_max)
        vy = jax.random.uniform(k2, (), minval=-cfg.vy_max, maxval=cfg.vy_max)
        vy = jnp.where(jax.random.uniform(k4, ()) < cfg.vy_zero_prob, 0.0, vy)
        hold_heading = jax.random.uniform(k5, ()) < cfg.yaw_zero_prob
        new_heading = jax.random.uniform(k3, (), minval=-jnp.pi, maxval=jnp.pi)
        heading_target = jnp.where(hold_heading, current_yaw, new_heading)
        cmd_xy = jnp.array([vx, vy], dtype=jnp.float32)
        zero = jax.random.uniform(k6, ()) < cfg.zero_cmd_prob
        cmd_xy = jnp.where(zero, jnp.zeros_like(cmd_xy), cmd_xy)
        heading_target = jnp.where(zero, current_yaw, heading_target)
        return cmd_xy.astype(jnp.float32), heading_target.astype(jnp.float32)

    def _command_from_heading(
        self, cmd_xy: jax.Array, heading_target: jax.Array, yaw: jax.Array
    ) -> jax.Array:
        cfg = self.config
        yaw_cmd = jnp.clip(
            cfg.heading_stiffness * self._wrap_to_pi(heading_target - yaw),
            -cfg.yaw_max,
            cfg.yaw_max,
        )
        return jnp.array([cmd_xy[0], cmd_xy[1], yaw_cmd], dtype=jnp.float32)

    def _gait_params(self, command: jax.Array) -> Tuple[jax.Array, jax.Array]:
        cfg = self.config
        speed = jnp.abs(command[0])
        t = jnp.clip(speed / jnp.maximum(cfg.vx_max, 1e-3), 0.0, 1.0)
        period = cfg.gait_period_slow * (1.0 - t) + cfg.gait_period_fast * t
        duty = cfg.duty_slow * (1.0 - t) + cfg.duty_fast * t
        return period, duty

    def _gait_clock(
        self, phase: jax.Array, command: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        _, duty = self._gait_params(command)
        phases = (phase + self._gait_offsets) % 1.0
        desired = (phases < duty).astype(jnp.float32)
        clock = jnp.stack(
            [jnp.sin(2.0 * jnp.pi * phases), jnp.cos(2.0 * jnp.pi * phases)],
            axis=1,
        ).reshape(-1)
        return clock.astype(jnp.float32), desired

    def _base_lin_vel_body(self, data: mjx.Data) -> jax.Array:
        R = data.xmat[self._base_body_id]
        return R.T @ data.qvel[0:3]

    def _gravity_body(self, quat_wxyz: jax.Array) -> jax.Array:
        w, x, y, z = quat_wxyz
        gx, gy, gz = 0.0, 0.0, -1.0
        return jnp.array(
            [
                gx * (1 - 2 * (y * y + z * z))
                + gy * (2 * (x * y - w * z))
                + gz * (2 * (x * z + w * y)),
                gx * (2 * (x * y + w * z))
                + gy * (1 - 2 * (x * x + z * z))
                + gz * (2 * (y * z - w * x)),
                gx * (2 * (x * z - w * y))
                + gy * (2 * (y * z + w * x))
                + gz * (1 - 2 * (x * x + y * y)),
            ],
            dtype=jnp.float32,
        )

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact if hasattr(data, "_impl") else data.contact
        geom = contact.geom
        dist = contact.dist
        active = dist < 0.0
        g0 = geom[:, 0]
        g1 = geom[:, 1]
        floor = self._floor_geom_id
        feet = self._foot_geom_ids

        def match_foot(foot_id):
            hit = jnp.logical_or(
                jnp.logical_and(g0 == floor, g1 == foot_id),
                jnp.logical_and(g1 == floor, g0 == foot_id),
            )
            return jnp.any(jnp.logical_and(hit, active)).astype(jnp.float32)

        return jax.vmap(match_foot)(feet)

    def _foot_positions_world(self, data: mjx.Data) -> jax.Array:
        return data.site_xpos[self._foot_site_ids]

    def _foot_positions_body(self, data: mjx.Data) -> jax.Array:
        R = data.xmat[self._base_body_id]
        base = data.xpos[self._base_body_id]
        feet = self._foot_positions_world(data)
        return (feet - base) @ R

    def _sample_hfield_z(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Bilinear sample of heightfield elevation (meters) at world (x, y)."""
        ncol = self._hfield_ncol
        nrow = self._hfield_nrow
        u = (x / self._hfield_half_x + 1.0) * 0.5 * (ncol - 1)
        v = (y / self._hfield_half_y + 1.0) * 0.5 * (nrow - 1)
        u = jnp.clip(u, 0.0, ncol - 1.0001)
        v = jnp.clip(v, 0.0, nrow - 1.0001)
        u0 = jnp.floor(u).astype(jnp.int32)
        v0 = jnp.floor(v).astype(jnp.int32)
        u1 = u0 + 1
        v1 = v0 + 1
        su = (u - u0.astype(jnp.float32)).astype(jnp.float32)
        sv = (v - v0.astype(jnp.float32)).astype(jnp.float32)
        elev = self._hfield_elev
        z00 = elev[v0, u0]
        z10 = elev[v0, u1]
        z01 = elev[v1, u0]
        z11 = elev[v1, u1]
        z = (
            (1.0 - su) * (1.0 - sv) * z00
            + su * (1.0 - sv) * z10
            + (1.0 - su) * sv * z01
            + su * sv * z11
        )
        return (z * self._hfield_elevation_z).astype(jnp.float32)

    def _terrain_height(self, x: jax.Array, y: jax.Array) -> jax.Array:
        if not self._has_hfield:
            return jnp.zeros_like(jnp.asarray(x, dtype=jnp.float32))
        x = jnp.asarray(x, dtype=jnp.float32)
        y = jnp.asarray(y, dtype=jnp.float32)
        if x.ndim == 0:
            return self._sample_hfield_z(x, y)
        return jax.vmap(self._sample_hfield_z)(x, y)

    def _sample_spawn_xy(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        rng, k_tile, k_off = jax.random.split(rng, 3)
        tile = jax.random.randint(k_tile, (), 0, N_TILES * N_TILES)
        tile_i = tile // N_TILES
        tile_j = tile % N_TILES
        x0 = -self._hfield_half_x + tile_j.astype(jnp.float32) * TILE_METERS
        y0 = -self._hfield_half_y + tile_i.astype(jnp.float32) * TILE_METERS
        offset = jax.random.uniform(k_off, (2,), minval=-0.6, maxval=0.6)
        x = x0 + 0.5 * TILE_METERS + offset[0]
        y = y0 + 0.5 * TILE_METERS + offset[1]
        return x.astype(jnp.float32), y.astype(jnp.float32)

    def _out_of_map(self, x: jax.Array, y: jax.Array) -> jax.Array:
        if not self._has_hfield:
            return jnp.array(False)
        return jnp.logical_or(
            jnp.abs(x) > self._hfield_half_x - MAP_EDGE_MARGIN,
            jnp.abs(y) > self._hfield_half_y - MAP_EDGE_MARGIN,
        )

    def _physics_invalid(self, data: mjx.Data) -> jax.Array:
        bad_qpos = jnp.any(jnp.isnan(data.qpos) | jnp.isinf(data.qpos))
        bad_qvel = jnp.any(jnp.isnan(data.qvel) | jnp.isinf(data.qvel))
        return jnp.logical_or(bad_qpos, bad_qvel)

    def _get_obs_frame(
        self,
        data: mjx.Data,
        command: jax.Array,
        phase: jax.Array,
        last_action: jax.Array,
        last_contacts: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        cfg = self.config
        lin = self._base_lin_vel_body(data).astype(jnp.float32)
        ang = data.qvel[3:6].astype(jnp.float32)
        grav = self._gravity_body(data.qpos[3:7])
        joint_pos = (self._leg_qpos(data) - self._default_pose).astype(jnp.float32)
        joint_vel = self._leg_qvel(data).astype(jnp.float32)
        clock, _ = self._gait_clock(phase, command)
        obs = jnp.concatenate(
            [
                command.astype(jnp.float32),
                lin,
                ang,
                grav,
                joint_pos,
                joint_vel,
                last_action.astype(jnp.float32),
                clock,
                last_contacts.astype(jnp.float32),
            ]
        )
        noise = jax.random.normal(rng, obs.shape, dtype=jnp.float32) * cfg.obs_noise_scale
        noise = noise.at[:3].set(0.0)
        return (obs + noise).astype(jnp.float32)

    def reset(self, rng: jax.Array) -> State:
        cfg = self.config
        rng, rng_cmd, rng_phase, rng_pose, rng_vel, rng_obs, rng_push = jax.random.split(
            rng, 7
        )
        data = mjx.make_data(self.mjx_model)
        qpos = self._home_qpos.at[self._leg_qposadr].add(
            jax.random.normal(rng_pose, (N_LEG,), dtype=jnp.float32) * 0.02
        )
        qvel = self._home_qvel.at[self._leg_dofadr].add(
            jax.random.normal(rng_vel, (N_LEG,), dtype=jnp.float32) * 0.05
        )
        if self._has_hfield:
            rng_push, rng_xy = jax.random.split(rng_push)
            spawn_x, spawn_y = self._sample_spawn_xy(rng_xy)
            spawn_z = self._terrain_height(spawn_x, spawn_y) + self._target_height
            qpos = qpos.at[0].set(spawn_x)
            qpos = qpos.at[1].set(spawn_y)
            qpos = qpos.at[2].set(spawn_z)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl)
        data = mjx.forward(self.mjx_model, data)

        yaw0 = self._yaw_from_quat(data.qpos[3:7])
        cmd_xy, heading_target = self._sample_velocity_command(rng_cmd, yaw0)
        command = self._command_from_heading(cmd_xy, heading_target, yaw0)
        phase = jax.random.uniform(rng_phase, ())

        last_action = jnp.zeros(ACTION_DIM, dtype=jnp.float32)
        applied = jnp.zeros(ACTION_DIM, dtype=jnp.float32)
        last_contacts = self._foot_contacts(data)
        last_foot_pos = self._foot_positions_body(data)
        last_joint_vel = self._leg_qvel(data).astype(jnp.float32)
        air_time = jnp.zeros(4, dtype=jnp.float32)

        frame = self._get_obs_frame(
            data, command, phase, last_action, last_contacts, rng_obs
        )
        history = jnp.tile(frame[None, :], (STACK_FRAMES, 1))
        obs = history.reshape(-1)

        info = {
            "rng": rng_push,
            "command": command,
            "cmd_xy": cmd_xy,
            "heading_target": heading_target,
            "phase": phase,
            "elapsed": jnp.array(0, dtype=jnp.int32),
            "steps_since_cmd": jnp.array(0, dtype=jnp.int32),
            "last_action": last_action,
            "applied_action": applied,
            "last_contacts": last_contacts,
            "last_foot_pos": last_foot_pos,
            "last_joint_vel": last_joint_vel,
            "air_time": air_time,
            "obs_history": history,
            "push_steps": jax.random.randint(
                rng_push, (), 1, cfg.push_interval_steps + 1
            ),
        }
        rel_h0 = (qpos[2] - self._terrain_height(qpos[0], qpos[1])).astype(jnp.float32)
        metrics = {
            "tracking_lin": jnp.array(0.0, dtype=jnp.float32),
            "tracking_ang": jnp.array(0.0, dtype=jnp.float32),
            "forward_vel": jnp.array(0.0, dtype=jnp.float32),
            "height": rel_h0,
            "tilt": jnp.array(0.0, dtype=jnp.float32),
            "heading": jnp.array(0.0, dtype=jnp.float32),
            "x_position": jnp.array(0.0, dtype=jnp.float32),
            "y_position": jnp.array(0.0, dtype=jnp.float32),
            "reward": jnp.array(0.0, dtype=jnp.float32),
        }
        return State(
            pipeline_state=data,
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(0.0, dtype=jnp.float32),
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        cfg = self.config
        data = state.pipeline_state
        info = state.info
        rng = info["rng"]
        rng, rng_cmd, rng_obs, rng_push, rng_next = jax.random.split(rng, 5)

        action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
        prev_applied = info["applied_action"]
        applied = (
            cfg.action_filter * prev_applied + (1.0 - cfg.action_filter) * action
        ).astype(jnp.float32)
        targets = self._default_pose + applied * self._action_scale
        ctrl = self._home_ctrl.at[:N_LEG].set(targets)
        data = data.replace(ctrl=ctrl)

        push_steps = info["push_steps"] - 1
        do_push = push_steps <= 0
        push = jax.random.uniform(
            rng_push, (2,), minval=-cfg.push_vel_xy, maxval=cfg.push_vel_xy
        )
        qvel = data.qvel
        qvel = jnp.where(do_push, qvel.at[0:2].add(push), qvel)
        data = data.replace(qvel=qvel)
        push_steps = jnp.where(
            do_push,
            jax.random.randint(
                rng_push, (), cfg.push_interval_steps // 2, cfg.push_interval_steps + 1
            ),
            push_steps,
        )

        def physics_step(d, _):
            d = mjx.step(self.mjx_model, d)
            return d, None

        data, _ = jax.lax.scan(physics_step, data, None, length=self._frame_skip)

        steps_since_cmd = info["steps_since_cmd"] + 1
        resample = steps_since_cmd >= cfg.cmd_resample_steps
        yaw = self._yaw_from_quat(data.qpos[3:7])
        new_xy, new_heading = self._sample_velocity_command(rng_cmd, yaw)
        cmd_xy = jnp.where(resample, new_xy, info["cmd_xy"])
        heading_target = jnp.where(resample, new_heading, info["heading_target"])
        command = self._command_from_heading(cmd_xy, heading_target, yaw)
        steps_since_cmd = jnp.where(resample, 0, steps_since_cmd)

        period, _ = self._gait_params(command)
        phase = (info["phase"] + self._dt / period) % 1.0
        elapsed = info["elapsed"] + 1

        reward, done, metrics, extras = self._reward(
            data=data,
            command=command,
            heading_target=heading_target,
            phase=phase,
            action=action,
            applied=applied,
            prev_applied=prev_applied,
            last_contacts=info["last_contacts"],
            last_foot_pos=info["last_foot_pos"],
            last_joint_vel=info["last_joint_vel"],
            air_time=info["air_time"],
        )

        contacts = extras["foot_contacts"]
        foot_pos = extras["foot_pos_body"]
        joint_vel = self._leg_qvel(data).astype(jnp.float32)
        air_time = extras["air_time"]

        frame = self._get_obs_frame(data, command, phase, applied, contacts, rng_obs)
        history = jnp.concatenate([info["obs_history"][1:], frame[None, :]], axis=0)
        obs = history.reshape(-1)

        new_info = {
            "rng": rng_next,
            "command": command,
            "cmd_xy": cmd_xy,
            "heading_target": heading_target,
            "phase": phase,
            "elapsed": elapsed,
            "steps_since_cmd": steps_since_cmd,
            "last_action": applied,
            "applied_action": applied,
            "last_contacts": contacts,
            "last_foot_pos": foot_pos,
            "last_joint_vel": joint_vel,
            "air_time": air_time,
            "obs_history": history,
            "push_steps": push_steps,
        }
        for k, v in info.items():
            if k not in new_info:
                new_info[k] = v

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward.astype(jnp.float32),
            done=done.astype(jnp.float32),
            metrics=metrics,
            info=new_info,
        )

    def _reward(
        self,
        *,
        data: mjx.Data,
        command: jax.Array,
        heading_target: jax.Array,
        phase: jax.Array,
        action: jax.Array,
        applied: jax.Array,
        prev_applied: jax.Array,
        last_contacts: jax.Array,
        last_foot_pos: jax.Array,
        last_joint_vel: jax.Array,
        air_time: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, Dict[str, jax.Array], Dict[str, jax.Array]]:
        cfg = self.config
        v_body = self._base_lin_vel_body(data)
        forward_vel = v_body[0]
        lateral_vel = v_body[1]
        vertical_vel = data.qvel[2]
        yaw_rate = data.qvel[5]
        ang_xy = jnp.sum(jnp.square(data.qvel[3:5]))
        height = data.qpos[2]
        terrain_z = self._terrain_height(data.qpos[0], data.qpos[1])
        rel_h = height - terrain_z
        grav = self._gravity_body(data.qpos[3:7])
        tilt = jnp.linalg.norm(grav[:2])

        contacts = self._foot_contacts(data)
        _, desired = self._gait_clock(phase, command)
        foot_pos = self._foot_positions_body(data)
        foot_world = self._foot_positions_world(data)
        foot_delta = (foot_pos - last_foot_pos) / jnp.maximum(self._dt, 1e-6)
        foot_speed_xy = jnp.linalg.norm(foot_delta[:, :2], axis=1)
        foot_terrain = self._terrain_height(foot_world[:, 0], foot_world[:, 1])
        foot_height = foot_world[:, 2] - foot_terrain

        lin_err = jnp.square(forward_vel - command[0]) + jnp.square(
            lateral_vel - command[1]
        )
        ang_err = jnp.square(yaw_rate - command[2])
        tracking_lin = jnp.exp(-lin_err / cfg.tracking_lin_sigma)
        tracking_ang = jnp.exp(-ang_err / cfg.tracking_ang_sigma)
        yaw = self._yaw_from_quat(data.qpos[3:7])
        heading_err = self._wrap_to_pi(heading_target - yaw)
        heading_r = jnp.exp(-jnp.square(heading_err) / cfg.heading_sigma)

        upright = jnp.exp(-8.0 * tilt * tilt)
        height_r = jnp.exp(-80.0 * (rel_h - self._target_height) ** 2)
        phase_r = 1.0 - jnp.mean(jnp.abs(contacts - desired))

        first_contact = (contacts > 0.5) * (last_contacts < 0.5)
        air_time = jnp.where(contacts > 0.5, 0.0, air_time + self._dt)
        air_time_r = jnp.sum(air_time * first_contact)

        swing = 1.0 - desired
        clearance = jnp.mean(swing * jnp.clip(foot_height - 0.05, -0.05, 0.05))

        slip = jnp.mean(contacts * jnp.clip(foot_speed_xy, 0.0, 5.0))
        action_rate = jnp.mean(jnp.square(applied - prev_applied))
        action_pen = jnp.mean(jnp.square(action))
        joint_vel = jnp.clip(self._leg_qvel(data), -40.0, 40.0)
        joint_vel_pen = jnp.mean(jnp.square(joint_vel))
        last_joint_vel_safe = jnp.clip(last_joint_vel, -40.0, 40.0)
        joint_acc = (joint_vel - last_joint_vel_safe) / jnp.maximum(self._dt, 1e-6)
        joint_acc_pen = jnp.mean(jnp.square(jnp.clip(joint_acc, -200.0, 200.0)))

        qj = self._leg_qpos(data)
        below = jnp.clip(self._joint_low - qj, 0.0, None)
        above = jnp.clip(qj - self._joint_high, 0.0, None)
        limit_pen = jnp.mean(jnp.square(below + above))

        fallen = jnp.logical_or(rel_h < cfg.fall_height, tilt > cfg.max_tilt)
        oob = self._out_of_map(data.qpos[0], data.qpos[1])
        invalid = self._physics_invalid(data)
        done = jnp.logical_or(jnp.logical_or(fallen, oob), invalid).astype(
            jnp.float32
        )

        reward = (
            cfg.tracking_lin_w * tracking_lin
            + cfg.tracking_ang_w * tracking_ang
            + cfg.heading_w * heading_r
            + cfg.upright_w * upright
            + cfg.height_w * height_r
            + cfg.phase_w * phase_r
            + cfg.air_time_w * air_time_r
            + cfg.clearance_w * clearance
            + cfg.alive_w
            - cfg.vertical_vel_w * jnp.square(vertical_vel)
            - cfg.ang_xy_w * ang_xy
            - cfg.action_rate_w * action_rate
            - cfg.action_w * action_pen
            - cfg.joint_vel_w * joint_vel_pen
            - cfg.joint_acc_w * joint_acc_pen
            - cfg.slip_w * slip
            - cfg.limit_w * limit_pen
            - cfg.terminate_w * jnp.maximum(done, fallen.astype(jnp.float32))
        )
        reward = jnp.clip(
            jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0
        )

        metrics = {
            "tracking_lin": tracking_lin,
            "tracking_ang": tracking_ang,
            "heading": jnp.abs(heading_err),
            "forward_vel": forward_vel,
            "height": rel_h,
            "tilt": tilt,
            "x_position": data.qpos[0],
            "y_position": data.qpos[1],
            "reward": reward,
        }
        extras = {
            "foot_contacts": contacts,
            "foot_pos_body": foot_pos,
            "air_time": air_time,
        }
        return reward, done, metrics, extras

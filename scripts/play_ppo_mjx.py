#!/usr/bin/env python3
"""Play a trained MJX PPO policy with classic MuJoCo viewer (no JAX physics)."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer
import numpy as np

from envs.mjx_spot_model import N_LEG, load_spot_mj_model
from envs.spot_locomotion_mjx import STACK_FRAMES, config_for_stage

MODELS_DIR = ROOT / "models"


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60, 60)))


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -60, 60)) - 1.0))


ACTIVATIONS = {
    "silu": silu,
    "swish": silu,
    "elu": elu,
    "relu": lambda x: np.maximum(x, 0),
}


def jax_to_numpy(x):
    if hasattr(x, "__array__"):
        return np.asarray(x)
    return x


class NumpyNormalizer:
    """Brax RunningStatisticsState → mean/std normalize."""

    def __init__(self, normalizer):
        if hasattr(normalizer, "mean"):
            mean = normalizer.mean
            std = getattr(normalizer, "std", None)
        elif isinstance(normalizer, dict):
            mean = normalizer["mean"]
            std = normalizer.get("std")
        else:
            raise TypeError(f"Unknown normalizer type: {type(normalizer)}")
        self.mean = np.asarray(jax_to_numpy(mean), dtype=np.float32).reshape(-1)
        if std is None:
            var = (
                normalizer["summed_variance"]
                if isinstance(normalizer, dict)
                else normalizer.summed_variance
            )
            std = np.sqrt(np.maximum(np.asarray(jax_to_numpy(var)), 1e-6))
        self.std = np.maximum(
            np.asarray(jax_to_numpy(std), dtype=np.float32).reshape(-1), 1e-6
        )

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.mean) / self.std).astype(np.float32)


class NumpyMLP:
    """Flax linen Dense MLP: params['params']['hidden_i']['kernel'|'bias']."""

    def __init__(self, params, activation="elu"):
        if "params" in params:
            params = params["params"]
        keys = sorted(params.keys(), key=lambda k: int(k.split("_")[-1]))
        self.layers = []
        for k in keys:
            layer = params[k]
            kernel = np.asarray(layer["kernel"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            self.layers.append((kernel, bias))
        self.activation = ACTIVATIONS[activation]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(self.layers):
            h = h @ w + b
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


def gravity_body(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    gx, gy, gz = 0.0, 0.0, -1.0
    return np.array(
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
        dtype=np.float32,
    )


def foot_contacts(model, data, floor_id, foot_ids) -> np.ndarray:
    contacts = np.zeros(4, dtype=np.float32)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 == floor_id:
            m = np.where(foot_ids == g2)[0]
        elif g2 == floor_id:
            m = np.where(foot_ids == g1)[0]
        else:
            continue
        if m.size:
            contacts[int(m[0])] = 1.0
    return contacts


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--stage", choices=("walk", "run"), default="walk")
    p.add_argument("--vx", type=float, default=None)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--timestep", type=float, default=0.004)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument(
        "--no-heading-hold",
        action="store_true",
        help="Disable yaw correction that holds the initial heading when wz=0",
    )
    p.add_argument("--heading-kp", type=float, default=1.25)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = (
        Path(args.model)
        if args.model
        else MODELS_DIR / f"ppo_spot_mjx_{args.stage}.pkl"
    )
    if not model_path.exists():
        alt = MODELS_DIR / "ppo_spot_mjx_walk.pkl"
        if model_path != alt and alt.exists():
            model_path = alt
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as f:
        ckpt = pickle.load(f)
    meta = ckpt.get("meta", {})
    timestep = float(meta.get("timestep", args.timestep))
    frame_skip = int(meta.get("frame_skip", args.frame_skip))
    stack_frames = int(meta.get("stack_frames", STACK_FRAMES))

    normalizer = NumpyNormalizer(ckpt["normalizer"])
    policy = NumpyMLP(ckpt["policy_params"], activation="elu")

    cfg = config_for_stage(args.stage)
    if args.vx is None:
        args.vx = 2.5 if args.stage == "run" else 0.6

    mj_model, ids = load_spot_mj_model(
        fast=True, timestep=timestep, frame_skip=frame_skip
    )
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, data, ids.home_key_id)
    mujoco.mj_forward(mj_model, data)

    default_pose = ids.default_pose.copy()
    action_scale = np.tile(
        [cfg.action_scale_hx, cfg.action_scale_hy, cfg.action_scale_knee], 4
    ).astype(np.float32)
    action_filter = float(cfg.action_filter)
    gait_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)
    command = np.array([args.vx, args.vy, args.wz], dtype=np.float32)
    heading_hold = (not args.no_heading_hold) and abs(args.wz) < 1e-6
    heading_target = quat_yaw(data.qpos[3:7])
    dt = ids.dt
    leg_qposadr = ids.leg_qposadr
    leg_dofadr = ids.leg_dofadr

    applied = np.zeros(N_LEG, dtype=np.float32)
    last_contacts = np.zeros(4, dtype=np.float32)
    phase = 0.0
    history = None

    def gait_period(vx: float) -> float:
        t = float(np.clip(abs(vx) / max(float(cfg.vx_max), 1e-3), 0.0, 1.0))
        return float(
            cfg.gait_period_slow * (1.0 - t) + cfg.gait_period_fast * t
        )

    def obs_frame() -> np.ndarray:
        R = data.xmat[ids.base_body_id].reshape(3, 3)
        lin = (R.T @ data.qvel[0:3]).astype(np.float32)
        ang = data.qvel[3:6].astype(np.float32)
        grav = gravity_body(data.qpos[3:7])
        joint_pos = (data.qpos[leg_qposadr] - default_pose).astype(np.float32)
        joint_vel = data.qvel[leg_dofadr].astype(np.float32)
        phases = (phase + gait_offsets) % 1.0
        clock = (
            np.stack(
                [np.sin(2 * np.pi * phases), np.cos(2 * np.pi * phases)], axis=1
            )
            .reshape(-1)
            .astype(np.float32)
        )
        return np.concatenate(
            [
                command,
                lin,
                ang,
                grav,
                joint_pos,
                joint_vel,
                applied,
                clock,
                last_contacts,
            ]
        ).astype(np.float32)

    def stacked_obs() -> np.ndarray:
        nonlocal history
        frame = obs_frame()
        if history is None:
            history = np.tile(frame[None, :], (stack_frames, 1))
        else:
            history = np.concatenate([history[1:], frame[None, :]], axis=0)
        return history.reshape(-1).astype(np.float32)

    def policy_action(obs: np.ndarray) -> np.ndarray:
        x = normalizer(obs)
        logits = policy(x)
        if logits.shape[-1] == 2 * N_LEG:
            mean = logits[:N_LEG]
        else:
            mean = logits
        return np.tanh(mean).astype(np.float32)

    def update_heading_command():
        if not heading_hold:
            return
        yaw = quat_yaw(data.qpos[3:7])
        err = wrap_to_pi(heading_target - yaw)
        command[2] = np.clip(args.heading_kp * err, -1.0, 1.0)

    print(f"Model: {model_path}")
    print(
        f"Command: vx={args.vx} vy={args.vy} wz={args.wz} "
        f"heading_hold={heading_hold}"
    )
    print("Controls: left-drag rotate, right-drag pan, scroll zoom, Esc quit")

    step_i = 0
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        while viewer.is_running():
            update_heading_command()
            obs = stacked_obs()
            action = policy_action(obs)
            applied = (
                action_filter * applied + (1.0 - action_filter) * action
            ).astype(np.float32)
            targets = default_pose + applied * action_scale
            data.ctrl[:N_LEG] = targets
            data.ctrl[N_LEG:] = 0.0
            for _ in range(frame_skip):
                mujoco.mj_step(mj_model, data)

            last_contacts = foot_contacts(
                mj_model, data, ids.floor_geom_id, ids.foot_geom_ids
            )
            phase = (phase + dt / gait_period(float(command[0]))) % 1.0
            viewer.sync()
            time.sleep(dt)

            step_i += 1
            if step_i % 50 == 0:
                R = data.xmat[ids.base_body_id].reshape(3, 3)
                v = float((R.T @ data.qvel[0:3])[0])
                print(
                    f"t={step_i*dt:6.1f}s  h={data.qpos[2]:.3f}  "
                    f"vx={v:.3f}  target={command[0]:.2f}  "
                    f"yaw={quat_yaw(data.qpos[3:7]):.2f}  wz={command[2]:.2f}",
                    flush=True,
                )
            if data.qpos[2] < 0.25:
                print("Fallen — resetting")
                mujoco.mj_resetDataKeyframe(mj_model, data, ids.home_key_id)
                mujoco.mj_forward(mj_model, data)
                applied[:] = 0
                history = None
                phase = 0.0
                heading_target = quat_yaw(data.qpos[3:7])
                command[2] = args.wz
            if args.max_steps and step_i >= args.max_steps:
                break


if __name__ == "__main__":
    main()

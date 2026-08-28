#!/usr/bin/env python3
"""Benchmark MJX throughput and Euler stability for Spot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from envs.mjx_spot_model import load_spot_mj_model


def bench_raw(
    *,
    n_envs: int,
    timestep: float,
    frame_skip: int,
    iterations: int,
    ls_iterations: int,
    steps: int = 100,
) -> dict:
    mj_model, ids = load_spot_mj_model(
        fast=True,
        timestep=timestep,
        frame_skip=frame_skip,
        iterations=iterations,
        ls_iterations=ls_iterations,
    )
    mx = mjx.put_model(mj_model)
    d = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, d, ids.home_key_id)
    mujoco.mj_forward(mj_model, d)

    dx = mjx.make_data(mx)
    dx = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape) if hasattr(x, "shape") else x,
        dx,
    )
    dx = dx.replace(
        qpos=jnp.broadcast_to(jnp.asarray(d.qpos), (n_envs, mj_model.nq)),
        ctrl=jnp.broadcast_to(jnp.asarray(d.ctrl), (n_envs, mj_model.nu)),
    )

    def one_ctrl_step(data):
        def body(dd, _):
            return mjx.step(mx, dd), None

        data, _ = jax.lax.scan(body, data, None, length=frame_skip)
        return data

    step_fn = jax.jit(jax.vmap(one_ctrl_step))
    t0 = time.time()
    dx = step_fn(dx)
    jax.block_until_ready(dx.qpos)
    compile_s = time.time() - t0

    t0 = time.time()
    for _ in range(steps):
        dx = step_fn(dx)
    jax.block_until_ready(dx.qpos)
    elapsed = time.time() - t0

    heights = np.asarray(dx.qpos[:, 2])
    env_sps = n_envs * steps / elapsed
    return {
        "n_envs": n_envs,
        "timestep": timestep,
        "frame_skip": frame_skip,
        "iterations": iterations,
        "ls_iterations": ls_iterations,
        "compile_s": compile_s,
        "env_steps_per_s": env_sps,
        "mean_height": float(heights.mean()),
        "min_height": float(heights.min()),
        "nan": bool(np.isnan(heights).any()),
        "stable": bool(heights.min() > 0.2 and not np.isnan(heights).any()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--timestep", type=float, nargs="+", default=[0.002, 0.004])
    p.add_argument("--frame-skip", type=int, nargs="+", default=[10, 5])
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--ls-iterations", type=int, default=5)
    p.add_argument("--steps", type=int, default=80)
    args = p.parse_args()

    print(f"jax devices: {jax.devices()}")
    print(
        f"{'n_envs':>7} {'dt':>6} {'fs':>3} {'it':>3} "
        f"{'compile':>8} {'env/s':>10} {'h_mean':>7} {'h_min':>7} ok"
    )
    best = None
    for dt in args.timestep:
        for fs in args.frame_skip:
            if abs(dt * fs - 0.02) > 1e-6:
                continue
            for n in args.n_envs:
                try:
                    r = bench_raw(
                        n_envs=n,
                        timestep=dt,
                        frame_skip=fs,
                        iterations=args.iterations,
                        ls_iterations=args.ls_iterations,
                        steps=args.steps,
                    )
                except Exception as e:
                    print(
                        f"{n:7d} {dt:6.3f} {fs:3d} {args.iterations:3d} "
                        f"FAILED: {type(e).__name__}: {str(e)[:80]}"
                    )
                    continue
                print(
                    f"{r['n_envs']:7d} {r['timestep']:6.3f} {r['frame_skip']:3d} "
                    f"{r['iterations']:3d} {r['compile_s']:8.1f} "
                    f"{r['env_steps_per_s']:10.0f} {r['mean_height']:7.3f} "
                    f"{r['min_height']:7.3f} {'Y' if r['stable'] else 'N'}"
                )
                if r["stable"] and (
                    best is None or r["env_steps_per_s"] > best["env_steps_per_s"]
                ):
                    best = r

    if best:
        print("\nBest stable config:")
        for k, v in best.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo stable config found.")


if __name__ == "__main__":
    main()

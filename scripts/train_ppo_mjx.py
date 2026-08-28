#!/usr/bin/env python3
"""Train Spot locomotion with MuJoCo MJX + Brax PPO (walk / run stages)."""

from __future__ import annotations

import argparse
import functools
import os
import pickle
import time
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

# Brax 0.14.x still calls jax.device_put_replicated, removed in JAX 0.11+.
if not hasattr(jax, "device_put_replicated"):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    def _device_put_replicated(value, devices):
        devices = tuple(devices)
        n = len(devices)

        def _add_axis(x):
            if x is None or isinstance(x, (str, bytes, bool)):
                return x
            try:
                x = jnp.asarray(x)
            except (TypeError, ValueError):
                return x
            return jnp.broadcast_to(x, (n,) + x.shape)

        value = jax.tree.map(_add_axis, value)
        mesh = Mesh(np.asarray(devices), axis_names=("i",))
        return jax.device_put(value, NamedSharding(mesh, P("i")))

    jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]

from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import flax.linen as linen

from envs.mjx_wrappers import make_domain_randomize_fn, wrap_for_training
from envs.spot_locomotion_mjx import SpotLocomotionEnv, config_for_stage

MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs" / "mjx"
MODELS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spot MJX PPO training")
    p.add_argument("--stage", choices=("walk", "run"), default="walk")
    p.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help=(
            "Env steps to train this run (default: 20M walk / 60M run). "
            "On resume this is additional steps; the log/pkl counter continues."
        ),
    )
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--episode-length", type=int, default=1000)
    p.add_argument("--unroll-length", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=32)
    p.add_argument("--num-updates-per-batch", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-cost", type=float, default=1e-2)
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clipping-epsilon", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--num-evals",
        type=int,
        default=20,
        help=(
            "Eval/checkpoint count. Capped automatically so training does not "
            "run extra PPO epochs beyond --timesteps."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=float, default=0.004)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--ls-iterations", type=int, default=5)
    p.add_argument("--no-domain-rand", action="store_true")
    p.add_argument("--no-obs-noise", action="store_true")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Brax Orbax checkpoint: parent *_ckpt dir (latest step) or a "
            "numbered step subdirectory such as .../000006225920"
        ),
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-checkpoint-path", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _resolve_checkpoint_path(path: str | Path) -> str:
    """Brax writes Orbax checkpoints under numbered step dirs, not the parent."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resume path does not exist: {p}")
    if p.is_file():
        raise FileNotFoundError(
            f"{p} is a file. Pass a Brax *_ckpt directory, not a .pkl policy."
        )
    if (p / "_CHECKPOINT_METADATA").exists():
        return str(p.resolve())

    step_dirs = [
        child
        for child in p.iterdir()
        if child.is_dir()
        and child.name.isdigit()
        and (child / "_CHECKPOINT_METADATA").exists()
    ]
    if not step_dirs:
        raise FileNotFoundError(
            f"No Brax Orbax checkpoint found under {p}. "
            "Expected numbered step folders like 000006225920."
        )
    latest = max(step_dirs, key=lambda d: int(d.name))
    return str(latest.resolve())


def _checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    return int(name) if name.isdigit() else 0


def _brax_schedule(
    num_timesteps: int,
    num_evals: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    action_repeat: int = 1,
) -> tuple[int, int, int]:
    """Match Brax PPO's epoch math and cap evals so we don't overshoot."""
    env_step = batch_size * unroll_length * num_minibatches * action_repeat
    min_epochs = max(1, int(np.ceil(num_timesteps / env_step)))
    capped_evals = min(num_evals, min_epochs + 1)
    epochs = max(capped_evals - 1, 1)
    steps_per_epoch = int(np.ceil(num_timesteps / (epochs * env_step)))
    planned = epochs * steps_per_epoch * env_step
    return capped_evals, env_step, planned


def _export_numpy_policy(params, path: Path, meta: dict) -> None:
    """Flatten normalizer + policy MLP weights for JIT-free playback."""
    normalizer, policy_params, value_params = params

    def to_numpy_tree(tree):
        def leaf(x):
            if hasattr(x, "shape"):
                return np.asarray(jax.device_get(x))
            if hasattr(x, "lo") and hasattr(x, "hi"):
                return {"hi": int(x.hi), "lo": int(x.lo)}
            if isinstance(x, (int, float, bool, str)) or x is None:
                return x
            try:
                return np.asarray(jax.device_get(x))
            except Exception:
                return x

        return jax.tree.map(leaf, tree)

    payload = {
        "normalizer": to_numpy_tree(normalizer),
        "policy_params": to_numpy_tree(policy_params),
        "value_params": to_numpy_tree(value_params),
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    args = parse_args()
    print(f"jax devices: {jax.devices()} backend={jax.default_backend()}")

    defaults = {"walk": 20_000_000, "run": 60_000_000}
    timesteps = args.timesteps
    if timesteps is None:
        timesteps = 200_000 if args.smoke else defaults[args.stage]
    if args.smoke:
        args.num_envs = min(args.num_envs, 512)
        args.num_evals = 5
        args.num_minibatches = min(args.num_minibatches, 8)
        args.unroll_length = min(args.unroll_length, 16)

    overrides = {}
    if args.no_obs_noise:
        overrides["obs_noise_scale"] = 0.0
    cfg = config_for_stage(args.stage, **overrides)

    env = SpotLocomotionEnv(
        config=cfg,
        fast=True,
        timestep=args.timestep,
        frame_skip=args.frame_skip,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
    )
    eval_env = SpotLocomotionEnv(
        config=config_for_stage(args.stage, obs_noise_scale=0.0, push_vel_xy=0.0),
        fast=True,
        timestep=args.timestep,
        frame_skip=args.frame_skip,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
    )

    out_prefix = args.out or f"ppo_spot_mjx_{args.stage}"
    ckpt_dir = (
        Path(args.save_checkpoint_path)
        if args.save_checkpoint_path
        else MODELS_DIR / f"{out_prefix}_ckpt"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    numpy_path = MODELS_DIR / f"{out_prefix}.pkl"

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        activation=linen.elu,
        distribution_type="tanh_normal",
        noise_std_type="scalar",
        init_noise_std=0.5,
    )

    randomization_fn = None if args.no_domain_rand else make_domain_randomize_fn()

    times = {"start": time.time(), "last": time.time()}
    restore_path = None
    resume_step = 0
    step_offset = 0
    train_timesteps = timesteps
    if args.resume:
        restore_path = _resolve_checkpoint_path(args.resume)
        resume_step = _checkpoint_step(restore_path)
        same_run = Path(restore_path).resolve().parent == ckpt_dir.resolve()
        if same_run:
            step_offset = resume_step
            print(
                f"resume={restore_path} from_step={resume_step} "
                f"additional={train_timesteps} "
                f"target={resume_step + train_timesteps}"
            )
        else:
            print(
                f"resume={restore_path} (new stage, this-run steps start at 0)"
            )

    num_evals, env_step, planned_steps = _brax_schedule(
        train_timesteps,
        args.num_evals,
        args.batch_size,
        args.unroll_length,
        args.num_minibatches,
    )
    if num_evals != args.num_evals:
        print(
            f"num_evals {args.num_evals} -> {num_evals} "
            f"(PPO epoch is {env_step} env-steps; extra evals would overshoot)"
        )

    def progress(num_steps, metrics):
        times["last"] = time.time()
        keys = [
            "eval/episode_reward",
            "eval/episode_reward_tracking_lin",
            "eval/episode_reward_tracking_ang",
            "eval/episode_heading",
            "eval/episode_reward_forward_vel",
            "eval/episode_reward_height",
            "eval/avg_episode_length",
            "training/entropy_loss",
            "training/policy_loss",
            "training/v_loss",
            "training/kl_mean",
            "training/clip_fraction",
            "training/sps",
        ]
        parts = [f"steps={int(num_steps) + step_offset}"]
        for k in keys:
            if k in metrics:
                parts.append(f"{k.split('/')[-1]}={float(metrics[k]):.4f}")
        if len(parts) == 1:
            for k, v in sorted(metrics.items()):
                if isinstance(v, (int, float)) or hasattr(v, "item"):
                    parts.append(f"{k}={float(v):.4f}")
                if len(parts) > 8:
                    break
        print(" | ".join(parts), flush=True)

    def policy_params_fn(current_step, make_policy, params):
        step = int(current_step) + step_offset
        meta = {
            "step": step,
            "stage": args.stage,
            "obs_dim": int(env.observation_size),
            "action_dim": int(env.action_size),
            "timestep": args.timestep,
            "frame_skip": args.frame_skip,
            "stack_frames": 3,
        }
        _export_numpy_policy(params, numpy_path, meta)
        snap = MODELS_DIR / f"{out_prefix}_step{step}.pkl"
        _export_numpy_policy(params, snap, meta)
        # Brax's own saver uses env_steps from 0; save here so resume
        # continues the numbered Orbax dirs instead of overwriting them.
        if int(current_step) > 0:
            ckpt_config = ppo_checkpoint.network_config(
                observation_size=int(env.observation_size),
                action_size=int(env.action_size),
                normalize_observations=True,
                network_factory=network_factory,
            )
            ppo_checkpoint.save(str(ckpt_dir), step, params, ckpt_config)

    print(
        f"stage={args.stage} timesteps={train_timesteps} "
        f"planned={planned_steps} num_evals={num_evals} "
        f"num_envs={args.num_envs} obs={env.observation_size} "
        f"act={env.action_size} dt={env.dt:.3f} "
        f"domain_rand={not args.no_domain_rand}"
    )

    make_policy, params, final_metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=train_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        action_repeat=1,
        wrap_env=True,
        wrap_env_fn=wrap_for_training,
        randomization_fn=randomization_fn,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        reward_scaling=1.0,
        clipping_epsilon=args.clipping_epsilon,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        normalize_advantage=True,
        bootstrap_on_timeout=True,
        network_factory=network_factory,
        seed=args.seed,
        num_evals=num_evals,
        num_eval_envs=min(128, args.num_envs),
        deterministic_eval=True,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        save_checkpoint_path=None,
        restore_checkpoint_path=restore_path,
        log_training_metrics=True,
        training_metrics_steps=max(
            train_timesteps // 50, args.num_envs * args.unroll_length
        ),
    )

    meta = {
        "step": step_offset + train_timesteps,
        "stage": args.stage,
        "obs_dim": int(env.observation_size),
        "action_dim": int(env.action_size),
        "timestep": args.timestep,
        "frame_skip": args.frame_skip,
        "stack_frames": 3,
        "final_metrics": {
            k: float(v)
            for k, v in final_metrics.items()
            if hasattr(v, "item") or isinstance(v, (int, float))
        },
    }
    _export_numpy_policy(params, numpy_path, meta)
    elapsed = time.time() - times["start"]
    print(f"Done in {elapsed/60:.1f} min. Saved {numpy_path}")
    print(f"Brax checkpoints: {ckpt_dir}")
    if final_metrics:
        for k in sorted(final_metrics):
            try:
                print(f"  {k}: {float(final_metrics[k]):.4f}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
